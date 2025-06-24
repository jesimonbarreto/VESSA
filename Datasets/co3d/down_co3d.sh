#!/bin/bash

# Target directory
DEST_DIR="/mnt/disks/stg_dataset/dataset/CO3D"
mkdir -p "$DEST_DIR"

# File containing filenames and download links
FILE_LIST="co3d_links.txt"

# Check if the list file exists
if [ ! -f "$FILE_LIST" ]; then
    echo "Error: File $FILE_LIST not found!"
    exit 1
fi

# Read the file line by line, skipping the header
sed 1d "$FILE_LIST" | while IFS=$'\t' read -r FILE_NAME URL; do
    if [ -f "$DEST_DIR/$FILE_NAME" ]; then
        echo "$FILE_NAME already exists. Skipping download."
    else
        echo "Downloading $FILE_NAME..."
        wget -c "$URL" -O "$DEST_DIR/$FILE_NAME"
    fi
done

echo "Download completed."
