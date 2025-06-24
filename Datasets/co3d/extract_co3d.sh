#!/bin/bash

# Directory containing the zip files
DATASET_DIR="/mnt/disks/stg_dataset/dataset/CO3D"

# Loop through all .zip files in the directory
for zip_file in "$DATASET_DIR"/*.zip; do
    # Skip if no .zip files are found
    [ -e "$zip_file" ] || continue

    echo "Extracting: $zip_file"

    # Extract the zip file into the dataset directory
    unzip -q "$zip_file" -d "$DATASET_DIR"

    # Check if extraction was successful
    if [ $? -eq 0 ]; then
        echo "Successfully extracted: $zip_file"
        rm "$zip_file"
        echo "Deleted: $zip_file"
    else
        echo "Failed to extract: $zip_file"
    fi
done
