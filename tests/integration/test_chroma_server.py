"""Chroma *server* mode (HttpClient): two app instances = two workers sharing one store.

Starts the `chroma` CLI bundled with chromadb on a free local port (no internet).
Skipped automatically if the CLI is unavailable or the server does not come up.
"""

import shutil
import socket
import subprocess
import time
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from rag_app.main import create_app
from tests.support import SAMPLE_DOC, ingest_files, make_settings

pytestmark = pytest.mark.chroma_server


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def chroma_server(tmp_path_factory):
    import sys
    cli = shutil.which("chroma") or str(Path(sys.executable).parent / "chroma")
    if not Path(cli).exists():
        pytest.skip("chroma CLI not available")
    port = _free_port()
    proc = subprocess.Popen([cli, "run", "--path", str(tmp_path_factory.mktemp("chroma-srv")),
                             "--host", "127.0.0.1", "--port", str(port)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(100):
            try:
                if httpx.get(f"http://127.0.0.1:{port}/api/v2/heartbeat", timeout=0.5).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(0.1)
        else:
            pytest.skip("chroma server did not start")
        yield port
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def test_two_workers_share_chroma_server(tmp_path, chroma_server):
    common = dict(chroma_host="127.0.0.1", chroma_port=chroma_server, collection_name="it_shared",
                  job_store_backend="sqlite", data_dir=tmp_path / "shared")
    # Same DATA_DIR => shared SQLite checkpointer + job store, like workers in one container.
    with TestClient(create_app(make_settings(tmp_path, **common))) as worker_a, \
         TestClient(create_app(make_settings(tmp_path, **common))) as worker_b:
        assert worker_a.get("/health").json()["vector_store"] == "chroma:http"
        r = worker_a.post("/ingest/files", files=[("files", (SAMPLE_DOC.name, SAMPLE_DOC.read_bytes(), "text/markdown"))])
        status_url = r.json()["status_url"]
        assert worker_b.get(status_url).json()["status"] == "succeeded"  # job visible to the other worker
        body = worker_b.post("/query", json={"question": "How long does the Atlas-7 battery last?",
                                             "session_id": "shared"}).json()
        assert "11 hours" in body["answer"]
        assert len(worker_a.get("/sessions/shared/history").json()["messages"]) == 2
        assert worker_b.delete(f"/documents/{SAMPLE_DOC.name}").status_code == 200
        assert worker_a.get("/documents").json()["total_chunks"] == 0
