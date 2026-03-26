"""
Download script for CSI-4CAST datasets.

This script downloads all available datasets from the CSI-4CAST Hugging Face organization
by checking for all possible combinations of channel models, delay spreads, and speeds.

Usage:
    python3 download.py [--output-dir OUTPUT_DIR]

If no arguments provided, it will download datasets to a 'datasets' folder.
"""

import argparse
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download
from tqdm import tqdm

# Configuration constants
ORG = "CSI-4CAST"

# Regular dataset parameters
LIST_CHANNEL_MODEL = ["A", "C", "D"]
LIST_DELAY_SPREAD = [30e-9, 100e-9, 300e-9]
LIST_MIN_SPEED = [1, 10, 30]

# Generalization dataset parameters
LIST_CHANNEL_MODEL_GEN = ["A", "B", "C", "D", "E"]
LIST_DELAY_SPREAD_GEN = [30e-9, 50e-9, 100e-9, 200e-9, 300e-9, 400e-9]
LIST_MIN_SPEED_GEN = sorted([*range(3, 46, 3), 1, 10])

def make_folder_name(cm: str, ds: float, ms: int, **kwargs) -> str:
    """Generate a standardized folder name based on channel model, delay spread, and minimum speed.

    Args:
        cm (str): Channel model identifier (e.g., 'A', 'B', 'C', 'D', 'E')
        ds (float): Delay spread in seconds (e.g., 30e-9, 100e-9, 300e-9)
        ms (int): Minimum speed in km/h (e.g., 1, 10, 30)
        **kwargs: Additional keyword arguments (unused)

    Returns:
        str: Formatted folder name in the format 'cm_{cm}_ds_{ds}_ms_{ms}'
             where ds is converted to nanoseconds and zero-padded to 3 digits,
             and ms is zero-padded to 3 digits

    Example:
        >>> make_folder_name('A', 30e-9, 10)
        'cm_A_ds_030_ms_010'
    """
    # the precision of the delay spread is int
    ds = round(ds * 1e9)
    ds_str = str(ds).zfill(3)

    # the precision of the min speed is .1
    ms_str = str(ms)
    ms_str = ms_str.zfill(3)

    # the file name
    return f"cm_{cm}_ds_{ds_str}_ms_{ms_str}"

def check_repo_exists(api: HfApi, repo_id: str) -> bool:
    """Check if a repository exists in the organization."""
    try:
        api.repo_info(repo_id, repo_type="dataset")
        return True
    except Exception:
        return False

def generate_dataset_combinations():
    """Generate all possible dataset combinations."""
    combinations = []
    
    # Stats dataset
    combinations.append("stats")
    
    # Train regular datasets
    for cm in LIST_CHANNEL_MODEL:
        for ds in LIST_DELAY_SPREAD:
            for ms in LIST_MIN_SPEED:
                folder_name = make_folder_name(cm, ds, ms)
                repo_name = f"train_regular_{folder_name}"
                combinations.append(repo_name)
    
    # Test regular datasets
    for cm in LIST_CHANNEL_MODEL:
        for ds in LIST_DELAY_SPREAD:
            for ms in LIST_MIN_SPEED:
                folder_name = make_folder_name(cm, ds, ms)
                repo_name = f"test_regular_{folder_name}"
                combinations.append(repo_name)
    
    # Test generalization datasets
    for cm in LIST_CHANNEL_MODEL_GEN:
        for ds in LIST_DELAY_SPREAD_GEN:
            for ms in LIST_MIN_SPEED_GEN:
                folder_name = make_folder_name(cm, ds, ms)
                repo_name = f"test_generalization_{folder_name}"
                combinations.append(repo_name)
    
    return combinations

def download_dataset(api: HfApi, org: str, repo_name: str, output_dir: Path, dry_run: bool = False) -> bool:
    """Download a single dataset if it exists."""
    repo_id = f"{org}/{repo_name}"
    
    if not check_repo_exists(api, repo_id):
        return False
    
    try:
        # Create target directory
        target_dir = output_dir / repo_name
        target_dir.mkdir(parents=True, exist_ok=True)
        
        if dry_run:
            # Create empty placeholder file
            placeholder_file = target_dir / "placeholder.txt"
            placeholder_file.write_text("")
            print(f"✅ Dry run - Created placeholder: {repo_name}")
        else:
            # Download the dataset
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                local_dir=target_dir,
                local_dir_use_symlinks=False
            )
            print(f"✅ Downloaded: {repo_name}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error downloading {repo_name}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Download all CSI-4CAST datasets from Hugging Face")
    parser.add_argument("--output-dir", "-o", default="datasets",
                       help="Output directory for downloaded datasets (default: 'datasets')")
    parser.add_argument("--dry-run", action="store_true",
                       help="Dry run mode: create empty placeholder files instead of downloading")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir).resolve()
    org = ORG
    
    mode = "Dry run" if args.dry_run else "Downloading"
    print(f"{mode} datasets from organization: {org}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize Hugging Face API
    api = HfApi()
    
    # Generate all possible combinations
    print("Generating dataset combinations...")
    combinations = generate_dataset_combinations()
    print(f"Total possible combinations: {len(combinations)}")
    print()
    
    # Download datasets
    action = "Checking and creating placeholders for" if args.dry_run else "Checking and downloading"
    print(f"{action} existing datasets...")
    downloaded_count = 0
    skipped_count = 0
    
    for repo_name in tqdm(combinations, desc="Processing datasets"):
        if download_dataset(api, org, repo_name, output_dir, args.dry_run):
            downloaded_count += 1
        else:
            skipped_count += 1
    
    print()
    if args.dry_run:
        print("🎉 Dry run complete!")
        print(f"✅ Created placeholders: {downloaded_count} datasets")
        print(f"⏭️  Skipped: {skipped_count} datasets (not found)")
        print(f"📁 Placeholders saved to: {output_dir}")
    else:
        print("🎉 Download complete!")
        print(f"✅ Downloaded: {downloaded_count} datasets")
        print(f"⏭️  Skipped: {skipped_count} datasets (not found)")
        print(f"📁 Datasets saved to: {output_dir}")
    print()
    print("To reconstruct the original folder structure, run:")
    print(f"python3 reconstruction.py --input-dir {output_dir}")

if __name__ == "__main__":
    main()