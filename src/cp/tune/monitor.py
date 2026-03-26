"""Launch Optuna dashboards for ablation studies.

Discovers every study.db under z_artifacts/outputs/ahpt/ablation/ and
launches an optuna-dashboard instance for each one on its own port.

Usage:
    python -m src.cp.tune.monitor                       # launch all dashboards
    python -m src.cp.tune.monitor --list-only            # just list discovered dbs
    python -m src.cp.tune.monitor --base-port 9000       # start ports at 9000
"""

import argparse
from pathlib import Path
import subprocess
import time


ABLATION_DIR = Path("z_artifacts") / "outputs" / "ahpt" / "ablation"


def discover_study_dbs(ablation_dir: Path) -> list[tuple[str, Path]]:
    """Return (human_label, db_path) for every study.db under *ablation_dir*.

    When multiple timestamped runs exist for the same ablation, only the
    most-recently-modified study.db is kept.
    """
    if not ablation_dir.exists():
        print(f"Directory does not exist: {ablation_dir}")
        return []

    best: dict[str, Path] = {}

    for db_path in sorted(ablation_dir.rglob("study.db")):
        rel = db_path.relative_to(ablation_dir)
        key = str(rel.parent.parent)  # category/name
        if key not in best or db_path.stat().st_mtime > best[key].stat().st_mtime:
            best[key] = db_path

    results = []
    for key in sorted(best):
        label = key.replace("/", " / ")
        results.append((label, best[key]))
    return results


def launch_dashboard(label: str, db_path: Path, port: int) -> subprocess.Popen | None:
    """Start one optuna-dashboard process for a study database."""
    storage_url = f"sqlite:///{db_path.absolute()}"
    cmd = ["optuna-dashboard", storage_url, "--port", str(port), "--host", "0.0.0.0"]

    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        time.sleep(2)
        if proc.poll() is None:
            return proc
        _, stderr = proc.communicate()
        print(f"  FAILED {label} (port {port})")
        if stderr:
            print(f"    stderr: {stderr.strip()}")
        return None
    except Exception as e:
        print(f"  ERROR launching {label}: {e}")
        return None


def main() -> None:
    """Parse arguments and launch dashboard processes."""
    parser = argparse.ArgumentParser(description="Launch optuna-dashboard for all ablation study.db files")
    parser.add_argument(
        "--ablation-dir",
        type=Path,
        default=ABLATION_DIR,
        help="Root ablation output directory",
    )
    parser.add_argument("--base-port", type=int, default=8080, help="Starting port number")
    parser.add_argument("--list-only", action="store_true", help="Only list discovered databases")
    args = parser.parse_args()

    study_dbs = discover_study_dbs(args.ablation_dir)

    if not study_dbs:
        print("No study.db files found.")
        return

    # --- list mode ---
    if args.list_only:
        print(f"Found {len(study_dbs)} study databases:\n")
        for i, (label, db_path) in enumerate(study_dbs):
            port = args.base_port + i
            print(f"  port {port}  {label}")
            print(f"           {db_path}\n")
        return

    # --- launch mode ---
    processes: list[tuple[str, int, subprocess.Popen]] = []

    print(f"Launching {len(study_dbs)} dashboards starting at port {args.base_port} ...\n")

    for i, (label, db_path) in enumerate(study_dbs):
        port = args.base_port + i
        proc = launch_dashboard(label, db_path, port)
        if proc:
            processes.append((label, port, proc))
            print(f"  OK  port {port}  {label}")
        else:
            print(f"  FAIL port {port}  {label}")

    if not processes:
        print("\nNo dashboards started.")
        return

    print("\n" + "=" * 60)
    print("RUNNING DASHBOARDS")
    print("=" * 60)
    for label, port, _ in processes:
        print(f"  http://localhost:{port}   {label}")
    print("=" * 60)
    print("Press Ctrl+C to stop all.\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping dashboards ...")
        for label, port, proc in processes:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
            print(f"  stopped port {port}  {label}")
        print("Done.")


if __name__ == "__main__":
    main()
