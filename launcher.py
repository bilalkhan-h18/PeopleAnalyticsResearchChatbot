"""Start and stop the chat app from an interactive console (Spyder, Jupyter, IPython).

    import launcher
    launcher.start()     # opens http://127.0.0.1:7860 in your browser
    launcher.stop()

Gradio's web server can't run inside Spyder's console: Spyder patches asyncio in a
way newer versions of uvicorn don't support. This runs app.py as a separate
Python process instead, so the console stays free and the server is unaffected.
"""

import os
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
_proc: subprocess.Popen | None = None


def _log_path() -> Path:
    from pa_chatbot.config import settings

    settings.data_dir.mkdir(parents=True, exist_ok=True)
    return settings.data_dir / "app.log"


def _is_up(url: str) -> bool:
    # Bypass any corporate proxy: this is a local address.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(url, timeout=2):
            return True
    except OSError:
        return False


def start(port: int = 7860, open_browser: bool = True, timeout: float = 120) -> str | None:
    """Launch the app in the background and return its URL once it responds."""
    global _proc
    url = f"http://127.0.0.1:{port}/"
    if _proc is not None and _proc.poll() is None:
        print(f"Already running at {url}")
        return url
    if _is_up(url):
        print(f"Something is already running at {url} - call launcher.stop(), or restart the kernel, then try again.")
        return None

    env = dict(os.environ, PORT=str(port), PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    # Keep local traffic away from proxies, which breaks Gradio's startup check.
    env["NO_PROXY"] = env["no_proxy"] = ",".join(filter(None, [env.get("NO_PROXY", ""), "localhost", "127.0.0.1"]))
    log = _log_path()
    log_fh = open(log, "w", encoding="utf-8")
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    _proc = subprocess.Popen(
        [sys.executable, str(ROOT / "app.py")],
        cwd=ROOT, env=env, stdout=log_fh, stderr=subprocess.STDOUT, creationflags=flags,
    )

    print("Starting the app (loading the index can take a little while)...", end="", flush=True)
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _proc.poll() is not None:
            print(f"\nThe app stopped during startup. Last lines of {log}:\n")
            print("".join(log.read_text(encoding="utf-8", errors="replace").splitlines(True)[-25:]))
            _proc = None
            return None
        if _is_up(url):
            print(f"\nRunning at {url}  (log: {log})")
            if open_browser:
                webbrowser.open(url)
            return url
        print(".", end="", flush=True)
        time.sleep(1)
    print(f"\nNo response after {timeout:.0f}s; check {log}. Stopping it.")
    stop()
    return None


def stop() -> None:
    global _proc
    if _proc is None or _proc.poll() is not None:
        print("The app isn't running from this console.")
        _proc = None
        return
    _proc.terminate()
    try:
        _proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        _proc.kill()
    _proc = None
    print("Stopped.")


def status() -> None:
    running = _proc is not None and _proc.poll() is None
    print("Running." if running else "Not running.")
