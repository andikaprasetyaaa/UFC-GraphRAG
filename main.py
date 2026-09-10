#!/usr/bin/env python3
"""
main.py
========
Pintu masuk ringkas untuk seluruh sistem.

    python main.py build              # bangun index (jalankan sekali)
    python main.py ask                # mode tanya-jawab interaktif
    python main.py ask "pertanyaan"   # mode sekali tanya
"""
from __future__ import annotations

import sys


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in {"build", "ask"}:
        print(__doc__)
        sys.exit(0 if len(sys.argv) < 2 else 1)

    command, rest = sys.argv[1], sys.argv[2:]
    sys.argv = [sys.argv[0], *rest]

    if command == "build":
        from scripts.build_index import main as build_main

        build_main()
    elif command == "ask":
        from scripts.ask import main as ask_main

        ask_main()


if __name__ == "__main__":
    main()
