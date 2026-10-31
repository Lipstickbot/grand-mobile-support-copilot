"""Local serving, readiness checks and streaming, atomic JSONL predictions."""

import argparse
import gzip
import json
import os
from pathlib import Path
import sys
import tempfile

from .config import Settings
from .schemas import TriageRequest, TriageResponse

MAX_ROW_BYTES = 128 * 1024


class _BatchError(Exception):
    """Messages are deliberately safe to display, never copied from input/errors."""


def _default_engine_factory(settings):
    from .engine import TriageEngine

    return TriageEngine(settings)


def _predict(engine, input_path: Path, output_path: Path):
    if input_path.resolve() == output_path.resolve():
        raise _BatchError("Input and output must be different files.")
    if output_path.exists() or output_path.is_symlink():
        raise _BatchError("Output already exists; choose a new output file.")
    temporary = None
    try:
        opener = gzip.open if input_path.suffix.lower() == ".gz" else open
        with opener(input_path, "rb") as source:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", newline="\n", dir=output_path.parent,
                prefix=".copilot-", suffix=".tmp", delete=False,
            ) as sink:
                temporary = Path(sink.name)
                line_number = 0
                while True:
                    try:
                        raw = source.readline(MAX_ROW_BYTES + 1)
                    except (OSError, EOFError):
                        raise _BatchError(f"Line {line_number + 1}: input could not be read.") from None
                    if not raw:
                        break
                    line_number += 1
                    if len(raw) > MAX_ROW_BYTES:
                        raise _BatchError(f"Line {line_number}: record exceeds 128 KiB.")
                    try:
                        request = TriageRequest.model_validate_json(raw)
                    except Exception:
                        raise _BatchError(f"Line {line_number}: invalid triage record.") from None
                    try:
                        response = TriageResponse.model_validate(engine.triage(request))
                        sink.write(response.model_dump_json() + "\n")
                    except Exception:
                        raise _BatchError(f"Line {line_number}: inference failed.") from None
                sink.flush()
                os.fsync(sink.fileno())
        # Windows rename refuses an existing destination. On POSIX, link provides
        # that same atomic no-clobber guarantee (rename would replace the output).
        # Both names are on the same filesystem because the temporary is a sibling.
        if os.name == "nt":
            os.rename(temporary, output_path)
        else:
            os.link(temporary, output_path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _parser():
    parser = argparse.ArgumentParser(description="Local support triage copilot")
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve", help="Serve the local HTTP API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    predict = commands.add_parser("predict", help="Predict JSONL or JSONL.gz records atomically")
    predict.add_argument("--input", "--in", dest="input", default=os.getenv("COPILOT_INPUT", os.getenv("INPUT")))
    predict.add_argument("--output", "--out", dest="output", default=os.getenv("COPILOT_OUTPUT", os.getenv("OUTPUT")))
    commands.add_parser("check", help="Load the engine and report readiness")
    return parser


def main(argv=None, *, engine_factory=None, settings=None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command == "predict" and (not args.input or not args.output):
        parser.error("predict requires --input and --output (or COPILOT_INPUT/COPILOT_OUTPUT)")
    engine = None
    exit_code = 0
    try:
        settings = settings if settings is not None else Settings.from_env()
        factory = engine_factory if engine_factory is not None else _default_engine_factory
        if args.command == "serve":
            from .api import create_app
            import uvicorn

            uvicorn.run(create_app(settings, factory), host=args.host, port=args.port)
            return 0
        if args.command == "predict":
            input_path, output_path = Path(args.input), Path(args.output)
            # Refuse destructive destinations before an expensive model load.
            if input_path.resolve() == output_path.resolve():
                raise _BatchError("Input and output must be different files.")
            if output_path.exists() or output_path.is_symlink():
                raise _BatchError("Output already exists; choose a new output file.")
        engine = factory(settings)
        engine.load()
        if args.command == "check":
            from .api import _is_ready

            health = engine.health()
            print(json.dumps(health, ensure_ascii=False))
            exit_code = 0 if _is_ready(health) else 1
        else:
            _predict(engine, input_path, output_path)
    except _BatchError as exc:
        print(str(exc), file=sys.stderr)
        exit_code = 1
    except Exception:
        print("Operation failed; check local configuration and engine readiness.", file=sys.stderr)
        exit_code = 1
    finally:
        if engine is not None:
            try:
                engine.close()
            except Exception:
                print("Engine shutdown failed.", file=sys.stderr)
                exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
