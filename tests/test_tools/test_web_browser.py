from __future__ import annotations

import pytest

from agentkb.tools.web_browser import WebBrowserTool


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1",
        "http://localhost",
        "http://169.254.169.254/latest/meta-data",
        "http://10.0.0.1",
        "http://[::1]",
        "https://user:password@example.com",
        "https://example.com:8443",
    ],
)
async def test_browser_rejects_private_or_credentialed_urls(url):
    with pytest.raises(ValueError):
        await WebBrowserTool._validate_public_url(url)
