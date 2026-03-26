"""
Reconstruction script for CSI-4CAST datasets.

This script helps users reconstruct the original folder structure after downloading
datasets from the CSI-4CAST Hugging Face organization.

Usage:
    python reconstruction.py [--input-dir INPUT_DIR] [--output-dir OUTPUT_DIR]

If no arguments provided, it will look for downloaded datasets in the current directory
and reconstruct the structure in a 'data' folder.
"""

import argparse
import shutil
from pathlib import Path


def create_directory_structure(base_path: Path):
    """Create the original directory structure"""
    dirs_to_create = [
        "stats",
        "test/regular", 
        "test/generalization",
        "train/regular"
    ]
    
    for dir_path in dirs_to_create:
        full_path = base_path / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {full_path}")

def find_downloaded_datasets(input_dir: Path):
    """Find all downloaded dataset folders"""
    datasets = {
        'stats': [],
        'test_regular': [],
        'test_generalization': [],
        'train_regular': []
    }
    
    # Look for folders that match our naming patterns
    for item in input_dir.iterdir():
        if item.is_dir():
            if item.name == "stats":
                datasets['stats'].append(item.name)
            elif item.name.startswith("test_regular_"):
                datasets['test_regular'].append(item.name)
            elif item.name.startswith("test_generalization_"):
                datasets['test_generalization'].append(item.name)
            elif item.name.startswith("train_regular_"):
                datasets['train_regular'].append(item.name)
    
    return datasets

def reconstruct_dataset(dataset_name: str, source_path: Path, target_path: Path, prefix_to_remove: str) -> bool:
    """Reconstruct a single dataset by removing prefix and moving to target location"""
    if prefix_to_remove:
        # Remove the prefix to get the original folder name
        original_name = dataset_name[len(prefix_to_remove):]
    else:
        original_name = dataset_name
    
    target_folder = target_path / original_name
    
    if target_folder.exists():
        print(f"Warning: {target_folder} already exists, skipping...")
        return False
    
    try:
        shutil.copytree(str(source_path), str(target_folder))
        print(f"Reconstructed: {dataset_name} -> {target_folder}")
        return True
    except Exception as e:
        print(f"Error reconstructing {dataset_name}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Reconstruct CSI-4CAST dataset folder structure")
    parser.add_argument("--input-dir", "-i", default=".", 
                       help="Directory containing downloaded datasets (default: current directory)")
    parser.add_argument("--output-dir", "-o", default="data",
                       help="Output directory for reconstructed structure (default: 'data')")
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    
    print(f"Looking for datasets in: {input_dir}")
    print(f"Reconstructing structure in: {output_dir}")
    print()
    
    # Create the directory structure
    create_directory_structure(output_dir)
    
    # Find all downloaded datasets
    datasets = find_downloaded_datasets(input_dir)
    
    total_reconstructed = 0
    
    # Reconstruct stats
    for dataset in datasets['stats']:
        source_path = input_dir / dataset
        target_path = output_dir / "stats"
        if reconstruct_dataset(dataset, source_path, target_path, ""):
            total_reconstructed += 1
    
    # Reconstruct test/regular datasets
    for dataset in datasets['test_regular']:
        source_path = input_dir / dataset
        target_path = output_dir / "test" / "regular"
        if reconstruct_dataset(dataset, source_path, target_path, "test_regular_"):
            total_reconstructed += 1
    
    # Reconstruct test/generalization datasets
    for dataset in datasets['test_generalization']:
        source_path = input_dir / dataset
        target_path = output_dir / "test" / "generalization"
        if reconstruct_dataset(dataset, source_path, target_path, "test_generalization_"):
            total_reconstructed += 1
    
    # Reconstruct train/regular datasets
    for dataset in datasets['train_regular']:
        source_path = input_dir / dataset
        target_path = output_dir / "train" / "regular"
        if reconstruct_dataset(dataset, source_path, target_path, "train_regular_"):
            total_reconstructed += 1
    
    print()
    print("✅ Reconstruction complete!")
    print(f"Total datasets reconstructed: {total_reconstructed}")
    print(f"Reconstructed structure available at: {output_dir}")
    print()
    print("Final structure:")
    print("data/")
    print("├── stats/")
    print("├── test/")
    print("│   ├── regular/")
    print("│   └── generalization/")
    print("└── train/")
    print("    └── regular/")

if __name__ == "__main__":
    main()