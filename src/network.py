"""
Network configuration helpers for Gaia archive access.

The preferred fix for TLS failures is to trust the corporate/root CA that signs
your HTTPS traffic. These helpers keep the learning notebooks usable in managed
networks where that CA is not available to Python.
"""

from __future__ import annotations

import http.client
import ssl
import warnings

import astropy.utils.data as aud
import requests
import urllib3


_REQUESTS_PATCHED = False
_ORIGINAL_REQUEST = requests.Session.request


def _unverified_http_context(http_version: int = 11) -> ssl.SSLContext:
    """
    Return an unverified context compatible with Python's http.client internals.

    Python's HTTPSConnection adds a couple of HTTP-specific TLS options around
    the base SSL context. Keeping those options avoids changing behavior beyond
    certificate verification.
    """
    context = ssl._create_unverified_context()
    if http_version == 11:
        context.set_alpn_protocols(["http/1.1"])
    if context.post_handshake_auth is not None:
        context.post_handshake_auth = True
    return context


def _disable_requests_verification() -> None:
    """
    Force requests sessions to skip TLS verification.

    Some astronomy libraries use requests instead of http.client. requests does
    not honor ssl._create_default_https_context, so it needs its own opt-out.
    """
    global _REQUESTS_PATCHED

    if _REQUESTS_PATCHED:
        return

    def request_without_tls_verification(self, method, url, **kwargs):
        kwargs["verify"] = False
        return _ORIGINAL_REQUEST(self, method, url, **kwargs)

    requests.Session.request = request_without_tls_verification
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    _REQUESTS_PATCHED = True


def configure_gaia_network(timeout: int = 60, *, verify_ssl: bool = False) -> None:
    """
    Apply network settings needed to reach the Gaia archive from notebooks.

    Args:
        timeout: Timeout in seconds for Astropy remote downloads.
        verify_ssl: Keep TLS certificate verification enabled. Set this to True
            when Python can already trust your network's CA bundle.
    """
    aud.REMOTE_TIMEOUT = timeout

    if verify_ssl:
        return

    warnings.warn(
        "SSL certificate verification is disabled for this Python session. "
        "Set SSL_CERT_FILE or REQUESTS_CA_BUNDLE to your corporate CA bundle "
        "and call configure_gaia_network(verify_ssl=True) when possible.",
        UserWarning,
        stacklevel=2,
    )

    # urllib and http.client users, including astroquery's Gaia TAP client.
    ssl._create_default_https_context = ssl._create_unverified_context
    if hasattr(http.client, "_create_https_context"):
        http.client._create_https_context = _unverified_http_context

    # requests users, including pyvo and some astroquery helpers.
    _disable_requests_verification()
