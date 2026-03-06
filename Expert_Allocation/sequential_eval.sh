#!/bin/bash

# Check if a backup name is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <backup_folder_name>"
  exit 1
fi

# Define the backup folder based on the argument
backup_dir="./evaluation_backup_$1"

# Define the list of evaluation Bash scripts
evaluation_scripts=(
  "eval_cola.sh"
  "eval_mrpc.sh"
  "eval_commonq.sh"
  "eval_openbook.sh"
  "eval_text_science_q.sh"
)

# Create a unique backup directory if it doesn't exist
if [ ! -d "$backup_dir" ]; then
  echo "Creating backup directory: $backup_dir"
  mkdir -p "$backup_dir"

  echo "Creating backups of evaluation scripts..."
  for script in "${evaluation_scripts[@]}"; do
    cp "$script" "$backup_dir/"
  done
  echo "Backup completed."
else
  echo "Backup directory already exists: $backup_dir. Running backed-up scripts."
fi

# Loop through each evaluation script and execute it from backup
echo "Executing backed-up evaluation scripts from $backup_dir..."
for script in "${evaluation_scripts[@]}"; do
  echo "Running $script..."
  bash "$backup_dir/$script"
  echo "$script completed."
done

echo "All evaluation scripts from $backup_dir have been executed."
