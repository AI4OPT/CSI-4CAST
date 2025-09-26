"""Unified analyzer for consolidated CSI prediction results handling both NMSE and SE analysis."""

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from src.utils.dirs import DIR_OUTPUTS
from src.utils.time_utils import get_current_time, get_latest_datetime_folder


def get_latest_results_path() -> Path | None:
    """Get the latest consolidated_results.csv from the gather directory.

    Returns:
        Path to the latest consolidated_results.csv file, or None if not found.

    """
    gather_dir = Path(DIR_OUTPUTS) / "testing" / "results" / "gather"

    if not gather_dir.exists():
        return None

    latest_dir = get_latest_datetime_folder(gather_dir)
    if latest_dir is None:
        return None

    results_file = latest_dir / "consolidated_results.csv"
    if not results_file.exists():
        return None

    return results_file


class CSIResultsAnalyzer:
    """Unified analyzer for consolidated CSI prediction results that handles both NMSE and SE analysis."""

    def __init__(self, results_path: Path | None = None):
        """Initialize the analyzer.

        Args:
            results_path: Path to consolidated_results.csv. If None, uses latest results.

        """
        if results_path is None:
            results_path = get_latest_results_path()

        if results_path is None:
            raise ValueError("No results path provided and no latest results found")

        print(f"Loading results from: {results_path}")
        self.df_raw = pd.read_csv(results_path)

        # Extract available models from data
        self.available_models = sorted(self.df_raw["model"].unique())

        print(f"Loaded {len(self.df_raw)} records")
        print(f"Available models: {self.available_models}")
        print(f"Available channels: {sorted(self.df_raw['cm'].unique())}")
        print(f"Available delay spreads: {sorted(self.df_raw['ds'].unique())}")
        print(f"Available min speeds: {sorted(self.df_raw['ms'].unique())}")
        print(f"Available is_gen: {sorted(self.df_raw['is_gen'].unique())}")
        print(f"Available scenarios: {sorted(self.df_raw['scenario'].unique())}")
        print(f"Available noise types: {sorted(self.df_raw['noise_type'].unique())}")
        print(f"Available pred_steps: {sorted(self.df_raw['pred_step'].unique())}")

    def _validate_columns(self, metric_type: Literal["nmse", "se"]):
        """Validate that required columns exist for the specified metric type."""
        base_columns = ["model", "cm", "ds", "ms", "is_gen", "scenario", "noise_type", "noise_degree", "pred_step"]

        if metric_type == "nmse":
            required_columns = [*base_columns, "nmse_mean"]
            metric_column = "nmse_mean"
        else:  # se
            required_columns = [*base_columns, "se_mean"]
            metric_column = "se_mean"

        missing_columns = [col for col in required_columns if col not in self.df_raw.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns for {metric_type} analysis: {missing_columns}")

        return metric_column

    def _get_scenario_category(self, row):
        """Categorize scenarios into regular, generalization, or robustness."""
        if row["is_gen"]:
            return "generalization"
        elif row["noise_type"] == "vanilla":
            return "regular"
        else:
            return "robustness"

    def create_pivot_table(self, metric_column: str) -> pd.DataFrame:
        """Create pivot table with pred_step aggregated (averaged) within scenarios.

        Args:
            metric_column: Column name for the metric (nmse_mean or se_mean).

        Returns:
            Pivot table with aggregated pred_step.

        """
        print("Creating pivot table...")
        df = self.df_raw.copy()

        # Aggregate pred_step by mean within scenarios
        print("Creating pivot table with pred_step aggregated...")
        df_aggregated = (
            df.groupby(["cm", "ds", "ms", "is_gen", "scenario", "noise_type", "noise_degree", "model"])[metric_column]
            .mean()
            .reset_index()
        )

        df_pivot_aggregated = df_aggregated.pivot_table(
            index=["cm", "ds", "ms", "is_gen", "scenario", "noise_type", "noise_degree"],
            columns="model",
            values=metric_column,
            aggfunc="first",
        ).reset_index()

        print(f"Pivot aggregated shape: {df_pivot_aggregated.shape}")

        return df_pivot_aggregated

    def calculate_model_rankings(
        self, df_pivot: pd.DataFrame, metric_column: str, higher_is_better: bool
    ) -> pd.DataFrame:
        """Calculate model rankings for each scenario row in the pivot table.

        Args:
            df_pivot: Pivot table with models as columns.
            metric_column: Column name for the metric values.
            higher_is_better: True for SE (higher is better), False for NMSE (lower is better).

        Returns:
            DataFrame with model rankings and scenario categorization.

        """
        print("Calculating model rankings...")
        df = df_pivot.copy()

        # Add scenario categorization
        df["scenario_category"] = df.apply(self._get_scenario_category, axis=1)

        # Calculate rankings for each scenario (row)
        model_columns = [col for col in df.columns if col in self.available_models]

        rankings_data = []
        for _, row in df.iterrows():
            # Get metric values for available models (non-null)
            metric_values = {col: row[col] for col in model_columns if pd.notna(row[col])}  # type: ignore

            if len(metric_values) == 0:
                continue

            # Sort by metric (ascending for NMSE, descending for SE)
            sorted_models = sorted(metric_values.items(), key=lambda x: float(x[1]), reverse=higher_is_better)

            # Create ranking dict (rank 1 = best, rank 2 = second best, etc.)
            rankings = {model: rank + 1 for rank, (model, _) in enumerate(sorted_models)}

            # Store scenario info + rankings
            scenario_info = {
                "cm": row["cm"],
                "ds": row["ds"],
                "ms": row["ms"],
                "is_gen": row["is_gen"],
                "scenario": row["scenario"],
                "noise_type": row["noise_type"],
                "noise_degree": row["noise_degree"],
                "scenario_category": row["scenario_category"],
            }

            # pred_step is aggregated, so not included in scenario info

            # Add rankings for each model
            for model in self.available_models:
                rankings_data.append(
                    {
                        **scenario_info,
                        "model": model,
                        "rank": rankings.get(model, np.nan),  # NaN if model not available for this scenario
                        f"{metric_column.replace('_mean', '')}_value": metric_values.get(model, np.nan),
                    }
                )

        return pd.DataFrame(rankings_data)

    def calculate_rank_distributions(self, df_rankings: pd.DataFrame) -> pd.DataFrame:
        """Calculate rank distributions for TDD/FDD combined with regular/generalization/robustness.

        Args:
            df_rankings: DataFrame with model rankings.

        Returns:
            DataFrame with rank distribution statistics.

        """
        print("Calculating rank distributions...")

        # Filter out rows where rank is NaN (model not available for that scenario)
        df_valid = df_rankings[df_rankings["rank"].notna()].copy()

        distributions = []

        # Group by scenario_category + scenario (TDD/FDD)
        grouped = df_valid.groupby(["scenario_category", "scenario"])
        for (scenario_category, scenario), group in grouped:  # type: ignore
            for model in self.available_models:
                model_data = group[group["model"] == model]

                if len(model_data) == 0:
                    continue

                ranks = model_data["rank"].values  # type: ignore
                total_scenarios = len(model_data)

                # Calculate rank distribution
                rank_counts = {}
                for rank in range(1, len(self.available_models) + 1):
                    rank_counts[f"rank_{rank}"] = int(np.sum(ranks == rank))  # type: ignore
                    rank_counts[f"rank_{rank}_pct"] = (
                        (np.sum(ranks == rank) / total_scenarios * 100) if total_scenarios > 0 else 0  # type: ignore
                    )

                # Calculate summary statistics
                distributions.append(
                    {
                        "scenario_category": scenario_category,
                        "scenario": scenario,  # TDD or FDD
                        "model": model,
                        "total_scenarios": total_scenarios,
                        "mean_rank": float(np.mean(ranks)) if len(ranks) > 0 else np.nan,  # type: ignore
                        "median_rank": float(np.median(ranks)) if len(ranks) > 0 else np.nan,  # type: ignore
                        "std_rank": float(np.std(ranks)) if len(ranks) > 0 else np.nan,  # type: ignore
                        "best_rank": float(np.min(ranks)) if len(ranks) > 0 else np.nan,  # type: ignore
                        "worst_rank": float(np.max(ranks)) if len(ranks) > 0 else np.nan,  # type: ignore
                        **rank_counts,
                    }
                )

        return pd.DataFrame(distributions)

    def analyze_metric(self, metric_type: Literal["nmse", "se"]) -> dict:
        """Analyze results for the specified metric type.

        Args:
            metric_type: Either "nmse" or "se".

        Returns:
            Dictionary containing analysis results.

        """
        print(f"=== ANALYZING {metric_type.upper()} RESULTS ===")

        # Validate columns and get metric column name
        metric_column = self._validate_columns(metric_type)
        higher_is_better = metric_type == "se"

        # Create pivot table with pred_step aggregated
        pivot_table = self.create_pivot_table(metric_column)

        # Calculate rankings and distributions
        rankings = self.calculate_model_rankings(pivot_table, metric_column, higher_is_better)
        distributions = self.calculate_rank_distributions(rankings)

        print(f"Total scenario-model combinations: {len(rankings)}")
        unique_scenarios = rankings.drop_duplicates(
            ["cm", "ds", "ms", "is_gen", "scenario", "noise_type", "noise_degree"]
        )
        print(f"Unique scenarios: {len(unique_scenarios)}")

        return {
            "metric_type": metric_type,
            "pivot_table": pivot_table,
            "rankings": rankings,
            "distributions": distributions,
        }

    def print_rank_summary(self, distributions: pd.DataFrame, approach_name: str, metric_type: str):
        """Print summary of rank distributions."""
        print(f"\n=== {approach_name} RANK SUMMARY ({metric_type.upper()}) ===")

        grouped_dist = distributions.groupby(["scenario_category", "scenario"])
        for (scenario_category, scenario), group in grouped_dist:  # type: ignore
            print(f"\n{scenario_category.upper()} - {scenario}:")
            for _, row in group.iterrows():
                print(
                    f"  {row['model']}: Mean rank = {row['mean_rank']:.2f}, "
                    f"Rank 1 = {row['rank_1_pct']:.1f}%, "
                    f"Total scenarios = {row['total_scenarios']}"
                )

        # Overall best models across all scenarios
        print(f"\n{approach_name} - OVERALL RANKINGS ({metric_type.upper()}):")
        overall_rankings = (
            distributions.groupby("model")
            .agg({"mean_rank": "mean", "total_scenarios": "sum", "rank_1": "sum"})
            .round(2)
        )
        overall_rankings["rank_1_pct"] = (overall_rankings["rank_1"] / overall_rankings["total_scenarios"] * 100).round(
            1
        )
        overall_rankings = overall_rankings.sort_values("mean_rank")  # type: ignore

        for model, row in overall_rankings.iterrows():
            print(
                f"  {model}: Mean rank = {row['mean_rank']:.2f}, "
                f"Rank 1 = {row['rank_1_pct']:.1f}%, "
                f"Total scenarios = {row['total_scenarios']}"
            )

    def save_results(self, results: dict, output_dir: Path | None = None):
        """Save analysis results to files.

        Args:
            results: Dictionary containing analysis results.
            output_dir: Directory to save results. If None, creates timestamped directory.

        """
        if output_dir is None:
            cur_time = get_current_time()
            metric_type = results["metric_type"]
            output_dir = Path(DIR_OUTPUTS) / "testing" / "results" / "analysis" / metric_type / cur_time

        output_path = output_dir if isinstance(output_dir, Path) else Path(str(output_dir))
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"Saving {results['metric_type'].upper()} results to: {output_path}")

        # Save pivot table
        results["pivot_table"].to_csv(output_path / "pivot_table.csv", index=False)
        print("- Saved pivot_table.csv")

        # Save rankings and distributions
        results["rankings"].to_csv(output_path / "rankings.csv", index=False)
        print("- Saved rankings.csv")

        results["distributions"].to_csv(output_path / "rank_distributions.csv", index=False)
        print("- Saved rank_distributions.csv")

        return output_path

    def run_analysis(self, metric_types: list[Literal["nmse", "se"]] | None = None, save_results: bool = True):
        """Run complete analysis for specified metric types.

        Args:
            metric_types: List of metric types to analyze ("nmse", "se", or both).
            save_results: Whether to save results to disk.

        """
        if metric_types is None:
            metric_types = ["nmse", "se"]
        print("=== RUNNING COMPLETE CSI RANKING ANALYSIS ===")

        for metric_type in metric_types:
            print(f"\n{'=' * 60}")
            print(f"ANALYZING {metric_type.upper()} METRICS")
            print(f"{'=' * 60}")

            try:
                # Run analysis for this metric type
                results = self.analyze_metric(metric_type)

                # Print summary
                self.print_rank_summary(results["distributions"], f"{metric_type.upper()} Analysis", metric_type)

                # Save results if requested
                if save_results:
                    output_path = self.save_results(results)
                    print(f"\n{metric_type.upper()} analysis completed and saved to: {output_path}")

            except Exception as e:
                print(f"Error analyzing {metric_type.upper()}: {e}")
                continue
