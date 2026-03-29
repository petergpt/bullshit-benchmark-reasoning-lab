#!/usr/bin/env python3
"""Build a read-only GitHub Pages snapshot for the Reasoning Lab studio."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from reasoning_lab_data import build_payload

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "dist" / "pages"
VIEWER_SRC = ROOT / "viewer" / "reasoning-annotation-studio.html"
VIEWER_OUT = OUT_DIR / "viewer" / "reasoning-annotation-studio.html"
BOOTSTRAP_OUT = OUT_DIR / "bootstrap.json"
ASSET_SRC = ROOT / "assets" / "bsbench.png"
ASSET_OUT = OUT_DIR / "assets" / "bsbench.png"
STORE_PATH = ROOT / "annotations" / "reasoning_lab.json"


def load_store() -> dict[str, Any]:
    with STORE_PATH.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object JSON in {STORE_PATH}")
    return payload


def inject_runtime_config(html: str) -> str:
    runtime = {
        "mode": "static",
        "readOnly": True,
        "staticBootstrapUrl": "../bootstrap.json",
    }
    snippet = (
        "<script>"
        f"window.REASONING_LAB_RUNTIME={json.dumps(runtime, ensure_ascii=False, separators=(',', ':'))};"
        "</script>\n"
    )
    if "</head>" not in html:
        raise ValueError("Expected </head> in viewer HTML.")
    return html.replace("</head>", f"{snippet}</head>", 1)


def write_index() -> None:
    index_html = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta http-equiv="refresh" content="0; url=./viewer/reasoning-annotation-studio.html">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Reasoning Lab</title>
</head>
<body>
  <p>Open <a href="./viewer/reasoning-annotation-studio.html">the Reasoning Lab viewer</a>.</p>
</body>
</html>
"""
    (OUT_DIR / "index.html").write_text(index_html, encoding="utf-8")


def main() -> None:
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    VIEWER_OUT.parent.mkdir(parents=True, exist_ok=True)
    ASSET_OUT.parent.mkdir(parents=True, exist_ok=True)

    lab_payload = build_payload()
    store = load_store()
    bootstrap = {
        "app": {
            "name": "BullshitBench Reasoning Lab",
            "mode": "static",
            "read_only": True,
            "read_only_message": "Read-only snapshot. Run the local server to edit labels.",
            "generated_at_utc": lab_payload.get("generated_at_utc"),
            "ai_labeling": {
                "enabled": False,
                "default_config_id": "",
                "configs": [],
                "max_concurrency": 0,
            },
        },
        "lab": lab_payload,
        "store": store,
    }

    BOOTSTRAP_OUT.write_text(
        json.dumps(bootstrap, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    VIEWER_OUT.write_text(
        inject_runtime_config(VIEWER_SRC.read_text(encoding="utf-8")),
        encoding="utf-8",
    )
    shutil.copy2(ASSET_SRC, ASSET_OUT)
    write_index()
    (OUT_DIR / ".nojekyll").write_text("", encoding="utf-8")

    size = BOOTSTRAP_OUT.stat().st_size / 1024 / 1024
    print(f"Wrote {BOOTSTRAP_OUT.relative_to(ROOT)} ({size:.2f} MiB)")
    print(f"Wrote {VIEWER_OUT.relative_to(ROOT)}")
    print(f"Wrote {ASSET_OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
