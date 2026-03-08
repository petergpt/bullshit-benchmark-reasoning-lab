#!/usr/bin/env python3
"""Build a local browser bundle for the GPT-5.4 reasoning atlas viewer."""

from __future__ import annotations

from gpt54_atlas_data import OUTPUT_PATH, ROOT, build_payload, write_browser_bundle


def main() -> None:
    payload = build_payload()
    write_browser_bundle(payload, OUTPUT_PATH)
    print(f"Wrote {OUTPUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
