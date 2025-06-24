#!/bin/bash

# Base URL from SharePoint (copied from the provided link)
BASE_URL="https://cuhko365.sharepoint.com/sites/GAP_Lab_MVImgNet/Shared%20Documents/MVImgNet_Release"

# Local directory to save the downloaded files
LOCAL_DIR="./MVImgNet_Release"

# Create local directory if it doesn't exist
mkdir -p "$LOCAL_DIR"

# Maximum number of attempts per file
MAX_RETRIES=5

# SharePoint credentials (if required)
USERNAME=""
PASSWORD="CUHKSZ-GapLab"

# Loop to download files from mvi_00.zip to mvi_42.zip
for i in $(seq -w 0 42); do
    FILE="mvi_${i}.zip"
    FILE_URL="$BASE_URL/$FILE"

    ATTEMPT=1

    while [ $ATTEMPT -le $MAX_RETRIES ]; do
        echo "Attempt $ATTEMPT to download $FILE..."

        # Download the file using wget with authentication
        wget --no-check-certificate --user="$USERNAME" --password="$PASSWORD" --output-document="$LOCAL_DIR/$FILE" "$FILE_URL"

        # Check if the file was downloaded successfully
        if [ -f "$LOCAL_DIR/$FILE" ]; then
            echo "$FILE downloaded successfully!"
            break
        else
            echo "Error downloading $FILE. Retrying..."
            ATTEMPT=$((ATTEMPT + 1))
            sleep 5
        fi
    done

    # If failed after max retries
    if [ $ATTEMPT -gt $MAX_RETRIES ]; then
        echo "Failed to download $FILE after $MAX_RETRIES attempts. Skipping to next file..."
    fi
done

echo "All downloads completed!"
