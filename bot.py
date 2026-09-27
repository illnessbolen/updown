"""Entry point. See README.md and `python bot.py --help`.

The previous v7 bot (REST polling + snipe/arb strategies) is kept unchanged in
legacy/bot_v7.py as the reference for the execution phase.
"""
from latarb.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
