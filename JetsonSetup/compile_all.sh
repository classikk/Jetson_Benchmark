#!/bin/bash

# Exit if no directory was given
if [ -z "$1" ]; then
    echo "Usage: $0 <directory>"
    exit 1
fi

BASE_DIR="$1"

# Find and run all compile.sh files
find "$BASE_DIR" -type f -name "compile.sh" | while read -r script; do
    echo "Running: $script"

    # Go to the script's directory
    script_dir=$(dirname "$script")
    (
        cd "$script_dir" || exit
        chmod +x compile.sh
        ./compile.sh
    )
done
