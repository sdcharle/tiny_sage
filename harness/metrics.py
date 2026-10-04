"""Approximate peak-memory sampling for local models.

On Apple Silicon the model weights live in unified memory owned by the llama-server process, so we
track (a) the RSS of llama-server processes and (b) system-wide used memory. Both are rough;
compare them across models, not against spec sheets.
"""

import threading

import psutil

SERVER_MARKERS = ("llama-server",)


def _server_rss() -> int:
    total = 0
    for p in psutil.process_iter(["name", "memory_info"]):
        name = (p.info["name"] or "").lower()
        if any(m in name for m in SERVER_MARKERS) and p.info["memory_info"]:
            total += p.info["memory_info"].rss
    return total


class MemorySampler:
    def __init__(self, interval_s: float = 0.5, enabled: bool = True):
        self.interval_s = interval_s
        self.enabled = enabled
        self.peak_proc = 0
        self.peak_sys = 0
        self._stop = threading.Event()
        self._thread = None

    def _sample(self):
        self.peak_proc = max(self.peak_proc, _server_rss())
        self.peak_sys = max(self.peak_sys, psutil.virtual_memory().used)

    def _loop(self):
        while not self._stop.wait(self.interval_s):
            self._sample()

    def __enter__(self):
        if self.enabled:
            self._sample()
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc):
        if self._thread:
            self._stop.set()
            self._thread.join()
            self._sample()

    def result(self) -> dict | None:
        if not self.enabled:
            return None
        mb = 1024 * 1024
        return {"peak_server_rss_mb": round(self.peak_proc / mb), "peak_system_used_mb": round(self.peak_sys / mb)}
