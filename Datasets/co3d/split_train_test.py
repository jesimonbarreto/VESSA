import os
import zipfile
import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split


# Set the seed to ensure reproducibility
SEED = 42
random.seed(SEED)

# Base path where the .zip files are located
base_path = '/mnt/disks/stg_dataset/dataset/CO3D/'

# Dictionaries to store video folder names per class
train_dict = defaultdict(list)
test_dict = defaultdict(list)

# Iterate through all .zip files in the directory
for filename in os.listdir(base_path):
    if filename.endswith('.zip') and filename.startswith('CO3D_'):
        zip_path = os.path.join(base_path, filename)
        with zipfile.ZipFile(zip_path, 'r') as zf:
            # Collect unique video folder names inside the zip
            folder_names = set()
            class_name = filename.replace('CO3D_', '').replace('.zip', '')
            for name in zf.namelist():
                parts = name.strip('/').split('/')
                if len(parts) >= 2:
                    folder_name = parts[1]
                    folder_names.add(folder_name)

            folder_names = sorted(folder_names)
            if not folder_names:
                continue

            # Split folders into train and test sets
            train_folders, test_folders = train_test_split(
                folder_names, test_size=0.25, random_state=SEED
            )

            train_dict[class_name].extend(train_folders)
            test_dict[class_name].extend(test_folders)

# Output file paths
train_path = os.path.join(base_path, 'train.npz')
test_path = os.path.join(base_path, 'test.npz')

# Save dictionaries as .npz files
np.savez(train_path, **train_dict)
np.savez(test_path, **test_dict)

print(f"Successfully saved to:\n{train_path}\n{test_path}")
