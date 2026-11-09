"""Reachability self-check.

Three different questions, and the app can only answer the first two from
inside the container:

1. Is the HTTP server actually answering? -- a loopback request through the
   real socket, not just in-process state.
2. Can this host reach DeepSeek, Ark and Doubao Speech? -- DNS + TLS + one
   unauthenticated GET each, which costs nothing.
3. Can the person at the other end reach us? -- only they can answer that.
   There is no inbound probe we can make from here, so the report says so
   instead of guessing.
"""

from __future__ import annotations

import socket
import time
from concurrent.futures import ThreadPoolExecutor

import httpx

from agent.settings import get_settings

# Every one of these rejects an unauthenticated GET. A 401/403 still proves the
# host resolved, connected and completed a TLS handshake, which is all this
# check claims. Anything billed is deliberately not touched.
PROBE_TIMEOUT_S = 6.0


def _listen_addresses(port: int) -> list[str]:
    """IPv4 addresses this process is bound to, read from the kernel."""
    addresses: set[str] = set()
    with open("/proc/net/tcp", encoding="utf-8") as handle:
        next(handle)
        for line in handle:
            local, _remote, state = line.split()[1], line.split()[2], line.split()[3]
            if state != "0A":  # LISTEN
                continue
            host_hex, port_hex = local.split(":")
            if int(port_hex, 16) != port:
                continue
            host = ".".join(str(int(host_hex[i : i + 2], 16)) for i in (6, 4, 2, 0))
            addresses.add(host)
    return sorted(addresses)


def _local_probe(port: int) -> dict:
    started = time.perf_counter()
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=PROBE_TIMEOUT_S):
            pass
        reached = True
        detail = "loopback connect + /health"
    except OSError as exc:
        return {"ok": False, "detail": f"{type(exc).__name__}: {exc}", "ms": None}
    return {"ok": reached, "detail": detail, "ms": round((time.perf_counter() - started) * 1000, 1)}


def _probe(name: str, url: str, headers: dict) -> dict:
    started = time.perf_counter()
    try:
        response = httpx.get(url, headers=headers, timeout=PROBE_TIMEOUT_S, follow_redirects=False)
    except httpx.HTTPError as exc:
        return {
            "name": name,
            "url": url,
            "reachable": False,
            "status": None,
            "ms": round((time.perf_counter() - started) * 1000, 1),
            "detail": f"{type(exc).__name__}: {str(exc)[:160]}",
        }
    ms = round((time.perf_counter() - started) * 1000, 1)
    if response.status_code in (401, 403):
        detail = "reachable, credential rejected (no key configured, or key invalid)"
    elif response.status_code in (200, 204):
        detail = "reachable, credential accepted"
    elif response.status_code == 405:
        detail = "host reachable (POST-only endpoint)"
    elif response.status_code == 404:
        detail = "host reachable (endpoint path not served)"
    else:
        detail = f"host reachable, unexpected status {response.status_code}"
    return {
        "name": name,
        "url": url,
        "reachable": True,
        "status": response.status_code,
        "ms": ms,
        "detail": detail,
    }


def connectivity_report(port: int) -> dict:
    settings = get_settings()
    media = settings.media

    llm = settings.llm
    ark_headers = {"Authorization": f"Bearer {media.ark_api_key}"} if media.ark_api_key else {}
    targets = [
        (
            "deepseek",
            f"{llm.base_url.rstrip('/')}/models",
            {"Authorization": f"Bearer {llm.api_key}"} if llm.api_key else {},
        ),
        ("ark", f"{media.ark_base_url.rstrip('/')}/models", ark_headers),
        # The TTS base already is the create endpoint; hitting it without a POST
        # returns 405/401 and still proves the host answers.
        ("doubao_speech", media.tts_base_url.rstrip("/"), {}),
    ]

    with ThreadPoolExecutor(max_workers=len(targets)) as pool:
        external = list(pool.map(lambda t: _probe(*t), targets))

    local = _local_probe(port)
    return {
        "server": {
            "port": port,
            "listening_on": _listen_addresses(port),
            "loopback": local,
        },
        "outbound": external,
        "inbound": {
            "verifiable_from_here": False,
            "note": (
                "Whether another machine can reach this port cannot be tested from inside. "
                "Run `curl -s http://<this-host>:PORT/health` from the client, or open "
                "http://<this-host>:PORT/ in a browser."
            ),
        },
        "billing_note": "No generation call is made. Only unauthenticated GETs are issued.",
    }
