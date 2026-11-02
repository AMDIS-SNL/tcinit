"""tcinit.network.use_env_proxy_for_aiohttp."""

import asyncio

import pytest

aiohttp = pytest.importorskip("aiohttp")

from tcinit.network import use_env_proxy_for_aiohttp  # noqa: E402


@pytest.fixture
def restore_session_init():
    init = aiohttp.ClientSession.__init__
    yield
    aiohttp.ClientSession.__init__ = init


def _trust_env(**kwargs):
    async def main():
        async with aiohttp.ClientSession(**kwargs) as session:
            return session.trust_env

    return asyncio.run(main())


def test_sessions_default_to_trust_env(restore_session_init):
    assert _trust_env() is False
    use_env_proxy_for_aiohttp()
    use_env_proxy_for_aiohttp()  # second call is a no-op
    assert _trust_env() is True
    assert _trust_env(timeout=aiohttp.ClientTimeout(total=5)) is True
    assert _trust_env(trust_env=False) is False
