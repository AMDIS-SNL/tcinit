"""Network helpers for running behind an HTTP proxy.

aiohttp ignores ``HTTP_PROXY`` / ``HTTPS_PROXY`` / ``NO_PROXY`` unless a
session is created with ``trust_env=True``. Libraries that build their own
``aiohttp.ClientSession()`` (earth2studio's NGC model downloader, for one)
then connect directly instead of through the proxy. On the Sandia network
that direct connection is intercepted and answered by the web filter's
block page, so the download fails with an unhelpful error such as
"Failed to get valid download URL for public model package".
"""

from __future__ import annotations

import functools


def use_env_proxy_for_aiohttp() -> None:
    """Make every new ``aiohttp.ClientSession`` default to ``trust_env=True``.

    Call once, before the library that downloads is used. Sessions that pass
    ``trust_env`` explicitly keep their own value. Calling it again is a no-op.
    """
    import aiohttp

    init = aiohttp.ClientSession.__init__
    if getattr(init, "_tcinit_trust_env", False):
        return

    @functools.wraps(init)
    def __init__(self, *args, trust_env=True, **kwargs):
        init(self, *args, trust_env=trust_env, **kwargs)

    __init__._tcinit_trust_env = True
    aiohttp.ClientSession.__init__ = __init__
