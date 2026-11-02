"""Create/reuse a Gemini File Search Store and synchronize local Markdown files."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
STORE_CONFIG_PATH = BASE_DIR / ".file_search_store.json"
MANIFEST_PATH = BASE_DIR / ".rag_upload_manifest.json"
MANAGED_PREFIX = "RAGDOC::"


def value(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def read_json(path: Path, fallback: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return fallback
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        print(f"[경고] {path.name} 파일을 읽지 못해 새 상태로 시작합니다.")
        return fallback


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY가 설정되지 않았습니다. backend/.env 파일을 확인하세요.")
    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=180_000),
    )


def get_or_create_store(client: genai.Client, requested_name: str | None) -> str:
    saved = read_json(STORE_CONFIG_PATH, {})
    store_name = (requested_name or os.getenv("FILE_SEARCH_STORE_NAME") or saved.get("store_name") or "").strip()
    if store_name:
        store = client.file_search_stores.get(name=store_name)
        print(f"기존 File Search Store 사용: {value(store, 'name', store_name)}")
        return value(store, "name", store_name)

    display_name = os.getenv("FILE_SEARCH_STORE_DISPLAY_NAME", "my-rag-knowledge-base")
    for store in client.file_search_stores.list():
        if value(store, "display_name", "") == display_name:
            store_name = value(store, "name", "")
            if store_name:
                print(f"동일한 표시 이름의 Store를 찾아 재사용합니다: {store_name}")
                write_json(STORE_CONFIG_PATH, {"store_name": store_name, "display_name": display_name})
                return store_name

    created = client.file_search_stores.create(
        config={
            "display_name": display_name,
            "embedding_model": os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-2"),
        }
    )
    store_name = value(created, "name", "")
    if not store_name:
        raise RuntimeError("Store를 만들었지만 Store 이름을 응답에서 확인하지 못했습니다.")
    write_json(STORE_CONFIG_PATH, {"store_name": store_name, "display_name": display_name})
    print(f"새 File Search Store 생성 완료: {store_name}")
    return store_name


def list_store_documents(client: genai.Client, store_name: str) -> list[Any]:
    return list(client.file_search_stores.documents.list(parent=store_name))


def delete_document(client: genai.Client, doc_name: str) -> None:
    client.file_search_stores.documents.delete(name=doc_name, config={"force": True})
    print(f"  기존 문서 삭제: {doc_name}")


def wait_for_operation(client: genai.Client, operation: Any, timeout_seconds: int = 900) -> Any:
    started = time.monotonic()
    while not value(operation, "done", False):
        if time.monotonic() - started > timeout_seconds:
            raise TimeoutError("파일 색인 작업이 15분 안에 완료되지 않았습니다.")
        time.sleep(3)
        operation = client.operations.get(operation)
    operation_error = value(operation, "error")
    if operation_error:
        raise RuntimeError(f"Google 파일 색인 작업 실패: {operation_error}")
    return operation


def sync_documents(
    client: genai.Client,
    store_name: str,
    docs_dir: Path,
    force: bool = False,
    prune: bool = False,
    dry_run: bool = False,
) -> tuple[int, int, int]:
    if not docs_dir.exists():
        raise FileNotFoundError(f"문서 폴더가 없습니다: {docs_dir}")

    files = sorted(p for p in docs_dir.rglob("*.md") if p.is_file() and not p.name.startswith("."))
    if not files:
        raise RuntimeError(f"{docs_dir} 폴더에서 .md 파일을 찾지 못했습니다.")

    manifest = read_json(MANIFEST_PATH, {"files": {}})
    manifest_files: dict[str, Any] = manifest.setdefault("files", {})
    remote_docs = list_store_documents(client, store_name)
    uploaded = skipped = deleted = 0
    present_paths: set[str] = set()

    for path in files:
        relative = path.relative_to(docs_dir).as_posix()
        present_paths.add(relative)
        display_name = f"{MANAGED_PREFIX}{relative}"
        digest = sha256_file(path)
        tracked = manifest_files.get(relative, {})
        matching_remote = [
            doc for doc in remote_docs
            if value(doc, "display_name", "") == display_name
        ]
        remote_names = {value(doc, "name", "") for doc in matching_remote}
        tracked_doc_name = tracked.get("document_name", "")

        if (
            not force
            and tracked.get("sha256") == digest
            and tracked_doc_name
            and tracked_doc_name in remote_names
        ):
            print(f"[변경 없음] {relative}")
            skipped += 1
            continue

        print(f"[업로드] {relative}")
        if dry_run:
            continue

        # Delete only after identifying this application's own display_name prefix.
        # Then upload the current content; the local manifest makes retries safe.
        for old_doc in matching_remote:
            old_name = value(old_doc, "name", "")
            if old_name:
                delete_document(client, old_name)
                deleted += 1

        operation = client.file_search_stores.upload_to_file_search_store(
            file=str(path),
            file_search_store_name=store_name,
            config={
                "display_name": display_name,
                "chunking_config": {
                    "white_space_config": {
                        "max_tokens_per_chunk": 400,
                        "max_overlap_tokens": 50,
                    }
                },
            },
        )
        wait_for_operation(client, operation)
        print("  색인 완료")
        uploaded += 1

        # Refresh the store listing to capture the new document resource name.
        remote_docs = list_store_documents(client, store_name)
        new_matches = [
            doc for doc in remote_docs
            if value(doc, "display_name", "") == display_name
        ]
        document_name = value(new_matches[-1], "name", "") if new_matches else ""
        manifest_files[relative] = {
            "sha256": digest,
            "display_name": display_name,
            "document_name": document_name,
        }
        write_json(MANIFEST_PATH, {"files": manifest_files})

    if prune and not dry_run:
        for doc in list_store_documents(client, store_name):
            display_name = value(doc, "display_name", "")
            if display_name.startswith(MANAGED_PREFIX):
                relative = display_name[len(MANAGED_PREFIX):]
                if relative not in present_paths:
                    doc_name = value(doc, "name", "")
                    if doc_name:
                        delete_document(client, doc_name)
                        deleted += 1
                    manifest_files.pop(relative, None)
        write_json(MANIFEST_PATH, {"files": manifest_files})

    return uploaded, skipped, deleted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="docs 폴더의 Markdown 파일을 Gemini File Search Store에 동기화합니다."
    )
    parser.add_argument("--docs", default=str(BASE_DIR / "docs"), help="Markdown 문서 폴더")
    parser.add_argument("--store-name", default=None, help="기존 Store resource name (생략 시 자동 재사용/생성)")
    parser.add_argument("--force", action="store_true", help="변경 여부와 관계없이 모든 파일 다시 업로드")
    parser.add_argument("--prune", action="store_true", help="로컬 docs에서 삭제된 관리 대상 문서도 Store에서 삭제")
    parser.add_argument("--dry-run", action="store_true", help="실제 업로드/삭제 없이 대상만 출력")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        client = make_client()
        store_name = get_or_create_store(client, args.store_name)
        uploaded, skipped, deleted = sync_documents(
            client=client,
            store_name=store_name,
            docs_dir=Path(args.docs).resolve(),
            force=args.force,
            prune=args.prune,
            dry_run=args.dry_run,
        )
        print("\n========== 작업 결과 ==========")
        print(f"업로드: {uploaded}개 / 변경 없음: {skipped}개 / 삭제: {deleted}개")
        print(f"FILE_SEARCH_STORE_NAME={store_name}")
        print(f"Store 설정 파일: {STORE_CONFIG_PATH}")
        print("Render 환경변수 FILE_SEARCH_STORE_NAME에 위 Store 이름을 입력하세요.")
        return 0
    except Exception as exc:
        print(f"\n[오류] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
