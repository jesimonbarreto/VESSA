#!/bin/bash

# Directory containing the zip files
DATASET_DIR="/mnt/disks/stg_dataset/dataset/CO3D"

# Loop through all zip files
for zip_file in "$DATASET_DIR"/*.zip; do
    # Check if any zip files exist
    [ -e "$zip_file" ] || continue

    echo "Extracting: $zip_file"

    # Unzip to the same directory
    unzip -q "$zip_file" -d "$DATASET_DIR"

    if [ $? -eq 0 ]; then
        echo "Successfully extracted: $zip_file"
        rm "$zip_file"
        echo "Deleted: $zip_file"
    else
        echo "Failed to extract: $zip_file"
    fi
done
