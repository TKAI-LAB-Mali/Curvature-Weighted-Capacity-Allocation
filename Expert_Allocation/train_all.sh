#!/bin/bash

# Check if a backup name is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <backup_folder_name>"
  exit 1
fi

# Define the backup folder based on the argument
backup_dir="./backup_scripts_$1"

# Define the list of Bash scripts
scripts=(
  "run_cola.sh"
  "run_mrpc.sh"
  "run_commonq.sh"
  "run_openbook.sh"
  "run_text_scienceq.sh"
  # run_all_10.sh
  # run_all_15.sh
  # run_all_20.sh
)

# Create a unique backup directory if it doesn't exist
if [ ! -d "$backup_dir" ]; then
  echo "Creating backup directory: $backup_dir"
  mkdir -p "$backup_dir"

  echo "Creating backups of scripts..."
  for script in "${scripts[@]}"; do
    cp "$script" "$backup_dir/"
  done
  echo "Backup completed."
else
  echo "Backup directory already exists: $backup_dir. Running backed-up scripts."
fi

# Loop through each script and execute the backed-up version
echo "Executing backed-up scripts from $backup_dir..."
for script in "${scripts[@]}"; do
  echo "Running $script..."
  bash "$backup_dir/$script"
  echo "$script completed."
done

echo "All scripts from $backup_dir have been executed."




#./run_all.sh run1 &
# ./run_all.sh run2 &
# ./run_all.sh run3 &