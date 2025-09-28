"""Performance table generation for model analysis.

This module creates comprehensive performance tables that present model rankings
and performance metrics in tabular format for easy comparison and analysis.
It supports both channel model and delay spread analysis across regular and
generalization conditions.

Key Features:
    - Channel model performance tables (A, C, D vs B, E)
    - Delay spread performance tables (regular vs generalization ranges)
    - Support for both SE and NMSE metrics
    - Automatic ranking and average calculations
    - Formatted output for both display and export
    - Separate analysis for TDD and FDD scenarios

Table Types:
    - se_cm: Spectral Efficiency across Channel Models
    - se_ds: Spectral Efficiency across Delay Spreads
    - nmse_cm: NMSE across Channel Models
    - nmse_ds: NMSE across Delay Spreads

Each table provides:
    - Individual condition performance values
    - Regular vs Generalization averages
    - Model rankings within each category
    - Statistical summaries and comparisons
"""

from pathlib import Path
from typing import Literal

import pandas as pd

# Import constants from data_utils
from src.utils.data_utils import (
    LIST_CHANNEL_MODEL,
    LIST_CHANNEL_MODEL_GEN,
    LIST_DELAY_SPREAD,
    LIST_DELAY_SPREAD_GEN,
    LIST_MIN_SPEED_TEST,
)


def create_scenario_table_cm(
    df_scenario: pd.DataFrame, scenario_name: str, metric: str = "se"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a performance table for channel models in a specific scenario (TDD or FDD).

    Args:
        df_scenario: Filtered dataframe for the scenario
        scenario_name: Name of the scenario (TDD or FDD)
        metric: Metric to use ('se' or 'nmse')

    Returns:
        Tuple of (display_df, raw_table)

    """
    # Define channel model order and grouping
    channel_order = LIST_CHANNEL_MODEL_GEN
    regular_channels = LIST_CHANNEL_MODEL
    generalization_channels = [cm for cm in LIST_CHANNEL_MODEL_GEN if cm not in LIST_CHANNEL_MODEL]

    # Use the appropriate metric column
    metric_column = f"{metric}_mean"

    # Group by model and channel model, calculate mean
    performance_table = df_scenario.groupby(["model", "cm"])[metric_column].mean().reset_index()
    print(f"  Aggregated to {len(performance_table)} model-channel combinations for {scenario_name}")

    # Pivot to create table format (models as rows, channels as columns)
    table = performance_table.pivot(index="model", columns="cm", values=metric_column)

    # Reorder columns according to specified order
    table = table.reindex(columns=channel_order)

    # Add group averages
    table["Regular_Avg"] = table[regular_channels].mean(axis=1)
    table["Generalization_Avg"] = table[generalization_channels].mean(axis=1)

    # Round values for better display
    table = table.round(6)

    # Create the display version with proper formatting
    display_data = []
    for model in table.index:
        row_data = {"Model": model}

        # Add regular channel performance
        for ch in regular_channels:
            row_data[f"{ch} (Regular)"] = f"{table.loc[model, ch]:.3f}" if not pd.isna(table.loc[model, ch]) else "N/A"

        # Add generalization channel performance
        for ch in generalization_channels:
            row_data[f"{ch} (Generalization)"] = (
                f"{table.loc[model, ch]:.3f}" if not pd.isna(table.loc[model, ch]) else "N/A"
            )

        # Add averages
        row_data["Regular Avg"] = (
            f"{table.loc[model, 'Regular_Avg']:.3f}" if not pd.isna(table.loc[model, "Regular_Avg"]) else "N/A"
        )
        row_data["Generalization Avg"] = (
            f"{table.loc[model, 'Generalization_Avg']:.3f}"
            if not pd.isna(table.loc[model, "Generalization_Avg"])
            else "N/A"
        )

        display_data.append(row_data)

    display_df = pd.DataFrame(display_data)

    # Sort by Regular Average (descending for SE, ascending for NMSE)
    ascending_order = metric == "nmse"  # NMSE: lower is better, SE: higher is better

    # Create sort values for proper sorting
    sort_values = []
    for val in display_df["Regular Avg"]:
        if val == "N/A":
            sort_values.append(999.0 if ascending_order else -999.0)
        else:
            try:
                sort_values.append(float(val))
            except ValueError:
                sort_values.append(999.0 if ascending_order else -999.0)

    # Sort by the computed values
    sorted_indices = sorted(range(len(sort_values)), key=lambda i: sort_values[i], reverse=not ascending_order)
    display_df = display_df.iloc[sorted_indices].reset_index(drop=True)

    # Also sort the raw table for consistent rankings
    table = table.sort_values("Regular_Avg", ascending=ascending_order)

    return display_df, table


def create_scenario_table_ds(
    df_scenario: pd.DataFrame, scenario_name: str, metric: str = "se"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a performance table for delay spreads in a specific scenario (TDD or FDD).

    Args:
        df_scenario: Filtered dataframe for the scenario
        scenario_name: Name of the scenario (TDD or FDD)
        metric: Metric to use ('se' or 'nmse')

    Returns:
        Tuple of (display_df, raw_table)

    """
    # Define delay spread order and grouping (similar to channel model structure)
    ds_order = LIST_DELAY_SPREAD_GEN
    regular_ds = LIST_DELAY_SPREAD  # LIST_DELAY_SPREAD - regular delay spreads
    generalization_ds = [ds for ds in LIST_DELAY_SPREAD_GEN if ds not in LIST_DELAY_SPREAD]

    # Create display names
    ds_display = regular_ds + generalization_ds
    ds_mapping = {ds: display for ds, display in zip(ds_order, ds_display, strict=False)}

    # Use the appropriate metric column
    metric_column = f"{metric}_mean"

    # Group by model and delay spread, calculate mean
    performance_table = df_scenario.groupby(["model", "ds"])[metric_column].mean().reset_index()
    print(f"  Aggregated to {len(performance_table)} model-delay_spread combinations for {scenario_name}")

    # Pivot to create table format (models as rows, delay spreads as columns)
    table = performance_table.pivot(index="model", columns="ds", values=metric_column)

    # Reorder columns according to specified order
    table = table.reindex(columns=ds_order)

    # Add group averages
    table["Regular_Avg"] = table[regular_ds].mean(axis=1)
    table["Generalization_Avg"] = table[generalization_ds].mean(axis=1)

    # Round values for better display
    table = table.round(6)

    # Create the display version with proper formatting
    display_data = []
    for model in table.index:
        row_data = {"Model": model}

        # Add regular delay spread performance
        for ds in regular_ds:
            display_name = ds_mapping[ds]
            row_data[f"{display_name} (Regular)"] = (
                f"{table.loc[model, ds]:.3f}" if not pd.isna(table.loc[model, ds]) else "N/A"  # type: ignore the pandas isna
            )

        # Add generalization delay spread performance
        for ds in generalization_ds:
            display_name = ds_mapping[ds]
            row_data[f"{display_name} (Generalization)"] = (
                f"{table.loc[model, ds]:.3f}" if not pd.isna(table.loc[model, ds]) else "N/A"  # type: ignore the pandas isna
            )

        # Add averages
        row_data["Regular Avg"] = (
            f"{table.loc[model, 'Regular_Avg']:.3f}" if not pd.isna(table.loc[model, "Regular_Avg"]) else "N/A"
        )
        row_data["Generalization Avg"] = (
            f"{table.loc[model, 'Generalization_Avg']:.3f}"
            if not pd.isna(table.loc[model, "Generalization_Avg"])
            else "N/A"
        )

        display_data.append(row_data)

    display_df = pd.DataFrame(display_data)

    # Sort by Regular Average (descending for SE, ascending for NMSE)
    ascending_order = metric == "nmse"  # NMSE: lower is better, SE: higher is better

    # Sort display_df by the sort values
    sort_values = []
    for val in display_df["Regular Avg"]:
        if val == "N/A":
            sort_values.append(999.0 if ascending_order else -999.0)  # Put N/A values at the end
        else:
            try:
                sort_values.append(float(val))
            except ValueError:
                sort_values.append(999.0 if ascending_order else -999.0)

    # Sort display_df by the sort values
    sorted_indices = sorted(range(len(sort_values)), key=lambda i: sort_values[i], reverse=not ascending_order)
    display_df = display_df.iloc[sorted_indices].reset_index(drop=True)

    # Also sort the raw table for consistent rankings
    table = table.sort_values("Regular_Avg", ascending=ascending_order)

    return display_df, table


def create_and_save_performance_tables(
    path_consolidated_results: Path,
    table_type: Literal["se_cm", "se_ds", "nmse_cm", "nmse_ds"],
    output_dir: Path | None = None,
) -> None:
    """Create and save performance tables for different metric and grouping combinations.

    Args:
        path_consolidated_results: Path to the consolidated results CSV file
        table_type: Type of table to create ('se_cm', 'se_ds', 'nmse_cm', 'nmse_ds')
        output_dir: Optional directory to save the tables

    """
    # Parse table type
    metric, grouping = table_type.split("_")

    # Read the consolidated results
    df = pd.read_csv(path_consolidated_results)

    # Filter data based on table type
    if grouping == "cm":
        # Channel model analysis - use regular channel model filtering
        df_filtered = df[
            (df["is_gen"])
            & (df["noise_type"] == "vanilla")
            & (df["ds"].isin(LIST_DELAY_SPREAD))
            & (df["ms"].isin(LIST_MIN_SPEED_TEST))
            & (df["cm"].isin(LIST_CHANNEL_MODEL_GEN))
        ].copy()

        filter_description = f"""Data: Filtered to match channel model analysis criteria:
  - is_gen = True (generalization data only)
  - noise_type = 'vanilla'
  - delay_spread in {LIST_DELAY_SPREAD} (ns)
  - min_speed in {LIST_MIN_SPEED_TEST}
  - channel_model in {LIST_CHANNEL_MODEL_GEN} (analyzing these)
Columns: A, C, D (Regular) | B, E (Generalization)"""
    else:  # grouping == "ds"
        # Delay spread analysis
        df_filtered = df[
            (df["is_gen"])
            & (df["noise_type"] == "vanilla")
            & (df["ds"].isin(LIST_DELAY_SPREAD_GEN))
            & (df["ms"].isin(LIST_MIN_SPEED_TEST))
            & (df["cm"].isin(LIST_CHANNEL_MODEL))
        ].copy()

        filter_description = f"""Data: Filtered for delay spread analysis:
  - is_gen = True (generalization data only)
  - noise_type = 'vanilla'
  - delay_spread in {LIST_DELAY_SPREAD_GEN} (ns) - analyzing these
  - min_speed in {LIST_MIN_SPEED_TEST} (regular)
  - channel_model in {LIST_CHANNEL_MODEL} (regular)
Columns: 30ns, 100ns, 300ns (Regular) | 50ns, 200ns, 400ns (Generalization)"""

    if df_filtered.empty:
        print(f"Warning: No data found matching {table_type} analysis criteria")
        return

    print(f"Filtered data: {len(df_filtered)} rows matching {table_type} analysis criteria")
    if grouping == "cm":
        print(f"Channel models: {sorted(df_filtered['cm'].unique().tolist())}")
        print(f"Delay spreads: {sorted(df_filtered['ds'].unique().tolist())}")
    else:
        print(f"Delay spreads: {sorted(df_filtered['ds'].unique().tolist())}")
        print(f"Channel models: {sorted(df_filtered['cm'].unique().tolist())}")
    print(f"Min speeds: {sorted(df_filtered['ms'].unique().tolist())}")

    # Create tables for each scenario
    scenarios = ["TDD", "FDD"]
    all_content = []

    for scenario in scenarios:
        df_scenario = df_filtered[df_filtered["scenario"] == scenario]

        if len(df_scenario) == 0:
            print(f"Warning: No data found for scenario {scenario}")
            continue

        print(f"\n{'=' * 60}")
        print(f"{metric.upper()} {grouping.upper()} Performance Table - {scenario} Scenario")
        print(f"{'=' * 60}")
        print(filter_description)
        print(f"Values: Averaged {metric.upper()} across all matching conditions")
        print("=" * 60)

        # Create table for this scenario
        if grouping == "cm":
            display_df, raw_table = create_scenario_table_cm(df_scenario, scenario, metric)  # type: ignore
        else:
            display_df, raw_table = create_scenario_table_ds(df_scenario, scenario, metric)  # type: ignore

        # Print to console using markdown format
        print(display_df.to_markdown(index=False))
        print("=" * 60)

        # Prepare content for file saving
        scenario_content = []
        scenario_content.append(f"{metric.upper()} {grouping.upper()} Performance Table - {scenario} Scenario")
        scenario_content.append("=" * 60)
        scenario_content.append(filter_description)
        scenario_content.append(f"Values: Averaged {metric.upper()} across all matching conditions")
        scenario_content.append("=" * 60)
        scenario_content.append("")
        scenario_content.append(display_df.to_markdown(index=False))
        scenario_content.append("")

        # Add model rankings
        if (
            hasattr(raw_table, "columns")
            and "Regular_Avg" in raw_table.columns
            and "Generalization_Avg" in raw_table.columns
        ):
            scenario_content.append("Model Rankings:")
            scenario_content.append("-" * 40)

            # Rank by Regular Average (descending for SE, ascending for NMSE)
            ascending_order = metric == "nmse"
            regular_rankings = raw_table.sort_values("Regular_Avg", ascending=ascending_order)

            if grouping == "cm":
                regular_desc = "By Regular Performance (A, C, D):"
                gen_desc = "By Generalization Performance (B, E):"
            else:
                regular_desc = "By Regular Performance (30ns, 100ns, 300ns):"
                gen_desc = "By Generalization Performance (50ns, 200ns, 400ns):"

            scenario_content.append(regular_desc)
            for i, (model, row) in enumerate(regular_rankings.iterrows(), 1):
                if not pd.isna(row["Regular_Avg"]):
                    scenario_content.append(f"  {i}. {model}: {row['Regular_Avg']:.3f}")

            scenario_content.append("")

            # Rank by Generalization Average
            gen_rankings = raw_table.sort_values("Generalization_Avg", ascending=ascending_order)
            scenario_content.append(gen_desc)
            for i, (model, row) in enumerate(gen_rankings.iterrows(), 1):
                if not pd.isna(row["Generalization_Avg"]):
                    scenario_content.append(f"  {i}. {model}: {row['Generalization_Avg']:.3f}")

            scenario_content.append("")

        all_content.extend(scenario_content)

        # Save individual CSV for this scenario
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save formatted table
            formatted_table_path = output_dir / f"{table_type}_performance_{scenario.lower()}_formatted.csv"
            display_df.to_csv(formatted_table_path, index=False)
            print(f"Formatted {scenario} table saved to: {formatted_table_path}")

    # Save combined text file with all tables
    if output_dir is not None and len(all_content) > 0:
        combined_table_path = output_dir / f"{table_type}_performance_tables.txt"
        with open(combined_table_path, "w") as f:
            f.write(f"{metric.upper()} {grouping.upper()} Performance Analysis\n")
            f.write("=" * 60 + "\n")
            f.write(f"Generated for {table_type} analysis:\n")
            f.write(filter_description.replace("Data: Filtered", "  Filtered") + "\n")
            f.write("Separate tables for TDD and FDD scenarios\n")
            f.write(
                f"Values {metric.upper()} averaged across all matching conditions per model-{grouping} combination\n"
            )
            f.write("=" * 60 + "\n\n")

            for line in all_content:
                f.write(line + "\n")

        print(f"\nCombined performance tables saved to: {combined_table_path}")

    print(f"\nPerformance table analysis completed for scenarios: {scenarios}")


def plot_table(
    path_consolidated_results: Path,
    output_dir: Path,
) -> None:
    """Generate comprehensive performance tables for all metric and grouping combinations.

    This is the main entry point for table generation, creating a complete set of
    performance comparison tables covering both SE and NMSE metrics across channel
    model and delay spread dimensions. It orchestrates the creation of all table
    types with consistent formatting and organization.

    Generated Tables:
        - SE Channel Model tables (se_cm): Performance across channel models A-E
        - SE Delay Spread tables (se_ds): Performance across delay spread ranges
        - NMSE Channel Model tables (nmse_cm): NMSE across channel models A-E
        - NMSE Delay Spread tables (nmse_ds): NMSE across delay spread ranges

    Each table type includes:
        - TDD and FDD scenario variants
        - Regular vs generalization comparisons
        - Model rankings and statistical summaries
        - Formatted CSV exports and text reports

    Parameters
    ----------
    path_consolidated_results : Path
        Path to consolidated_results.csv containing all experimental data.
    output_dir : Path
        Root output directory for table organization.

    Output Structure
    ----------------
    output_dir/
    ├── se_cm/           # SE across channel models
    ├── se_ds/           # SE across delay spreads
    ├── nmse_cm/         # NMSE across channel models
    └── nmse_ds/         # NMSE across delay spreads

    Notes
    -----
    This function provides a comprehensive tabular analysis complement to the
    graphical visualizations, enabling detailed numerical comparison of model
    performance across different experimental dimensions. Tables are formatted
    for both human readability and further statistical analysis.

    """
    dir_se_cm = output_dir / "se_cm"
    dir_se_cm.mkdir(parents=True, exist_ok=True)
    dir_se_ds = output_dir / "se_ds"
    dir_se_ds.mkdir(parents=True, exist_ok=True)
    dir_nmse_cm = output_dir / "nmse_cm"
    dir_nmse_cm.mkdir(parents=True, exist_ok=True)
    dir_nmse_ds = output_dir / "nmse_ds"
    dir_nmse_ds.mkdir(parents=True, exist_ok=True)

    create_and_save_performance_tables(path_consolidated_results, "se_cm", dir_se_cm)
    create_and_save_performance_tables(path_consolidated_results, "se_ds", dir_se_ds)
    create_and_save_performance_tables(path_consolidated_results, "nmse_cm", dir_nmse_cm)
    create_and_save_performance_tables(path_consolidated_results, "nmse_ds", dir_nmse_ds)

    print(f"Performance tables saved to: {output_dir}")
