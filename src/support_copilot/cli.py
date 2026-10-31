"""Entry point used by `make predict` / `make serve` and the `copilot` console script."""
from __future__ import annotations

import argparse
import os
import sys

from .v3.__main__ import main as v3_main


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Grand Mobile Support Copilot")
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve", help="HTTP service + admin UI")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    predict = sub.add_parser("predict", help="JSONL in -> JSONL out")
    predict.add_argument("--input", "--in", dest="input", default=os.getenv("COPILOT_INPUT"))
    predict.add_argument("--output", "--out", dest="output", default=os.getenv("COPILOT_OUTPUT"))
    sub.add_parser("train", help="train models from .private/data")
    args = parser.parse_args(argv)
    if args.command == "predict":
        if not args.input or not args.output:
            parser.error("predict requires --input and --output")
        v3_main(["predict", "--input", args.input, "--output", args.output])
    elif args.command == "serve":
        v3_main(["serve", "--host", args.host, "--port", str(args.port)])
    else:
        v3_main(["train"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
