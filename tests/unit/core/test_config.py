from rag_app.core.config import Settings
from tests.support import ROOT


def test_env_example_parses_and_empty_secrets_are_none():
    s = Settings(_env_file=ROOT / ".env.example")
    assert s.llm_model == "claude-sonnet-5-5"
    assert s.anthropic_api_key is None and s.openai_api_key is None and s.chroma_host is None
    assert s.chroma_mode == "embedded"


def test_chroma_server_mode_and_paths(tmp_path):
    s = Settings(_env_file=None, chroma_host="chroma", data_dir=tmp_path)
    assert s.chroma_mode == "http"
    assert s.resolved_checkpoint_db == tmp_path / "checkpoints.sqlite"
    assert s.resolved_jobs_db == tmp_path / "jobs.sqlite"


def test_extensions_normalised():
    assert Settings(_env_file=None, allowed_extensions=["PDF", ".Md"]).allowed_extensions == [".pdf", ".md"]
