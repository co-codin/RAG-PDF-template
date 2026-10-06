from pydantic import SecretStr

from rag_app.core.security import verify_api_key


def test_auth_disabled_when_no_key():
    assert verify_api_key(None, None)


def test_key_checks():
    key = SecretStr("s3cret")
    assert verify_api_key(key, "s3cret")
    assert not verify_api_key(key, "wrong")
    assert not verify_api_key(key, None)
