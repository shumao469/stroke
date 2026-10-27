"""Small dependency-free command-line entry point; imports stay lazy."""
import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(prog="strokeweave", description="Synthetic-only auditable research prototype")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("info", help="Print capability and safety boundaries")
    demo = commands.add_parser("demo", help="Run the fixed five generated technical scenarios")
    demo.add_argument("--state-dir", default="runtime/demo")
    serve = commands.add_parser("serve", help="Launch the local English workbench")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8801)
    serve.add_argument("--state-dir", default="runtime")
    args = parser.parse_args(argv)
    if args.command == "info":
        from . import __version__
        print(json.dumps({
            "name": "StrokeWeave", "version": __version__,
            "execution_mode": "generated-input research prototype",
            "patient_execution": False, "training_enabled": False,
            "labels_sealed": ["CAL", "HOLD", "test"],
            "planning": "rule-based structured routing, not an LLM",
            "external_model_plugins": "disabled; no weights distributed",
            "controller_superiority": "not established",
            "clinical_use": "not permitted by this research implementation",
        }, indent=2))
        return 0
    if args.command == "demo":
        from .workflow import run_generated_demo
        result = run_generated_demo(args.state_dir)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0 if result["complete"] == 2 and result["referred"] == 3 else 1
    if args.command == "serve":
        from .app import main as app_main
        return app_main(["--host", args.host, "--port", str(args.port), "--state-dir", str(Path(args.state_dir))])
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
