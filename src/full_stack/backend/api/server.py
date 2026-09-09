"""Dashboard server entry point."""

from __future__ import annotations

import threading
import time
import webbrowser

from .paths import WEB_CLIENT_DIR


def serve_dashboard(host: str = "127.0.0.1", port: int = 5005, open_browser: bool = True) -> None:
    """Run the dashboard until interrupted."""
    import uvicorn

    from .app import create_app

    url = f"http://{host if host != '0.0.0.0' else 'localhost'}:{port}"
    built = (WEB_CLIENT_DIR / "index.html").exists()

    lines = ["", "  COMPASS dashboard", f"  {url}"]
    if not built:
        lines += [
            "",
            "  The web client is not built yet. Build it once with:",
            "    npm --prefix src/full_stack/frontend install",
            "    npm --prefix src/full_stack/frontend run build",
            "",
            "  Or run the client in dev mode alongside this server:",
            "    npm --prefix src/full_stack/frontend run dev",
        ]
    lines.append("")
    # Flush so the URL appears immediately, even when stdout is redirected.
    print("\n".join(lines), flush=True)

    if open_browser and built:
        def launch() -> None:
            time.sleep(1.2)
            try:
                webbrowser.open(url)
            except Exception:
                pass

        threading.Thread(target=launch, daemon=True).start()

    uvicorn.run(create_app(), host=host, port=port, log_level="warning", access_log=False)


def main() -> None:
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Run the COMPASS dashboard server")
    parser.add_argument("--host", default=os.getenv("COMPASS_UI_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("COMPASS_UI_PORT", "5005")))
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    serve_dashboard(host=args.host, port=args.port, open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
