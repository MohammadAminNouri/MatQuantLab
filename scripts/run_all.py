from __future__ import annotations

import argparse
import subprocess
import sys


def run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd))
    subprocess.check_call(cmd)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the complete MatQuantLab pipeline.")
    parser.add_argument("--demo", action="store_true", help="Use synthetic demo data for testing only.")
    args = parser.parse_args()

    update_cmd = [sys.executable, "scripts/run_daily_update.py"]
    if args.demo:
        update_cmd.append("--demo")
    run(update_cmd)
    run([sys.executable, "scripts/generate_research_report.py"])


if __name__ == "__main__":
    main()
