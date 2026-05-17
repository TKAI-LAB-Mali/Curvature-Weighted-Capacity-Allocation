import argparse
from pathlib import Path

# The exact generic filenames generated before the naming update
OLD_FILES = {
    "weight_norm_interactive.html",
    "weight_norms_before.json",
    "weight_norms_after.json",
    "activation_norms_before.json",
    "activation_norms_after.json",
    "weight_norm_before_after.png",
    "activation_norm_before_after.png",
    "weight_norm_method_overlay.png",
    "activation_norm_method_overlay.png",
    "weight_norm_delta_heatmap.png",
    "activation_norm_delta_heatmap.png",
    "activation_norm_ratio_overlay_magnitude.png",
    "activation_norm_ratio_overlay_sparsegpt.png",
    "activation_norm_ratio_overlay_wanda.png",
    "weight_norm_ratio_overlay_magnitude.png",
    "weight_norm_ratio_overlay_sparsegpt.png",
    "weight_norm_ratio_overlay_wanda.png",
}

def clean_directory(target_dir: str, dry_run: bool):
    root_path = Path(target_dir)
    
    if not root_path.exists() or not root_path.is_dir():
        print(f"[ERROR] Directory not found: {root_path}")
        return

    print(f"\nScanning {root_path} for old artifacts...")
    
    # Recursively find all files in the directory tree
    all_files = root_path.rglob("*")
    
    files_to_delete = []
    for file_path in all_files:
        if file_path.is_file():
            # Check if the filename exactly matches one of our old targets
            if file_path.name in OLD_FILES:
                files_to_delete.append(file_path)

    if not files_to_delete:
        print("No old files found! Your directory is clean.")
        return

    print(f"Found {len(files_to_delete)} ghost files.\n")

    for f in files_to_delete:
        if dry_run:
            print(f"[DRY RUN - WOULD DELETE]: {f}")
        else:
            f.unlink()
            print(f"[DELETED]: {f}")

    if dry_run:
        print("\n[INFO] This was a DRY RUN. No files were actually deleted.")
        print("Run again with --force to permanently delete these files.")
    else:
        print("\n[SUCCESS] Cleanup complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Purge old generic plot/norm files.")
    parser.add_argument("--dir", type=str, default="depth_prior_results",
                        help="Target root directory to scan (e.g., depth_prior_results or results)")
    parser.add_argument("--force", action="store_true",
                        help="Disable dry-run and actually delete the files.")
    
    args = parser.parse_args()
    
    # By default, dry_run is True unless --force is passed
    clean_directory(args.dir, dry_run=not args.force)