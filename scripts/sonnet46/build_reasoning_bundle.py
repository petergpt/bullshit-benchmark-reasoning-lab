#!/usr/bin/env python3
"""Build the local Sonnet 4.6 reasoning browser bundle."""

from __future__ import annotations

from reasoning_data import OUTPUT_PATH, ROOT, build_payload, write_browser_bundle


def main() -> None:
    payload = build_payload()
    write_browser_bundle(payload, OUTPUT_PATH)
    print(f"Wrote {OUTPUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
