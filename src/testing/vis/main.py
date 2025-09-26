import argparse
import datetime
import shutil
from pathlib import Path

from src.testing.vis.line import plot_line
from src.testing.vis.radar import plot_radar
from src.testing.vis.table import plot_table
from src.testing.vis.violin import plot_violin
from src.utils.dirs import DIR_OUTPUTS
from src.utils.time_utils import get_current_time, get_latest_datetime_folder


def vis_main(
    input_path_consolidated_results: Path | None = None,
    input_dir_analysis_nmse: Path | None = None,
    input_dir_analysis_se: Path | None = None,
    input_dir_computational_overhead: Path | None = None,
):
    dir_testing = Path(DIR_OUTPUTS) / "testing"
    dir_computational_overhead = dir_testing / "computational_overhead"
    dir_results = dir_testing / "results"
    dir_analysis = dir_results / "analysis"
    dir_gather = dir_results / "gather"

    dir_vis = dir_testing / "vis" / get_current_time()
    dir_vis.mkdir(parents=True, exist_ok=True)

    if input_path_consolidated_results is None:
        latest_gather_dir = get_latest_datetime_folder(dir_gather)
        if latest_gather_dir is None:
            raise ValueError("No gather directory found")
        path_consolidated_results = latest_gather_dir / "consolidated_results.csv"
        if not path_consolidated_results.exists():
            raise FileNotFoundError(f"Consolidated results file not found: {path_consolidated_results}")

    if input_dir_analysis_nmse is None:
        latest_analysis_nmse_dir = get_latest_datetime_folder(dir_analysis / "nmse")
        if latest_analysis_nmse_dir is None:
            raise ValueError("No analysis_nmse directory found")
        dir_analysis_nmse = latest_analysis_nmse_dir

    if input_dir_analysis_se is None:
        latest_analysis_se_dir = get_latest_datetime_folder(dir_analysis / "se")
        if latest_analysis_se_dir is None:
            raise ValueError("No analysis_se directory found")
        dir_analysis_se = latest_analysis_se_dir

    if input_dir_computational_overhead is None:
        latest_computational_overhead_dir = get_latest_datetime_folder(dir_computational_overhead)
        if latest_computational_overhead_dir is None:
            raise ValueError("No computational_overhead directory found")
        dir_computational_overhead = latest_computational_overhead_dir

    # plot line
    dir_line = dir_vis / "line"
    dir_line.mkdir(parents=True, exist_ok=True)

    print("#####################")
    print("### Plotting line ###")
    print("#####################")
    plot_line(path_consolidated_results, dir_line)

    # plot radar
    dir_radar = dir_vis / "radar"
    dir_radar.mkdir(parents=True, exist_ok=True)

    print("######################")
    print("### Plotting radar ###")
    print("######################")
    plot_radar(dir_computational_overhead, dir_analysis_se, dir_analysis_nmse, dir_radar)

    # plot violin
    dir_violin = dir_vis / "violin"
    dir_violin.mkdir(parents=True, exist_ok=True)

    print("#######################")
    print("### Plotting violin ###")
    print("#######################")
    plot_violin(dir_analysis_se, dir_analysis_nmse, dir_violin)

    # plot table
    dir_table = dir_vis / "table"
    dir_table.mkdir(parents=True, exist_ok=True)

    print("####################")
    print("### Plotting table ###")
    print("####################")
    plot_table(path_consolidated_results, dir_table)


if __name__ == "__main__":
    vis_main()
