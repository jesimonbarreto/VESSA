import os
import random
import numpy as np
import sys

# Dataset path
dataset_path = '/mnt/disks/stg_dataset/dataset/mvimgnet/data/'

# Set the seed to ensure reproducibility
SEED = 42
random.seed(SEED)

# Simple progress bar display function
def print_progress(current, total, prefix="Progress", length=50):
    progress = int(length * current / total)
    bar = f"[{'#' * progress}{'.' * (length - progress)}]"
    sys.stdout.write(f"\r{prefix}: {bar} {current}/{total}")
    sys.stdout.flush()

# Function to list all videos by class
def get_video_list():
    video_dict = {}
    classes = os.listdir(dataset_path)
    total_classes = len(classes)
    
    for i, class_name in enumerate(classes, start=1):
        class_path = os.path.join(dataset_path, class_name)
        if os.path.isdir(class_path):
            video_dict[class_name] = []
            for video in os.listdir(class_path):
                video_path = os.path.join(class_path, video, 'images')
                if os.path.isdir(video_path):  # Check if 'images' folder exists
                    video_dict[class_name].append(video)
        # Update progress bar
        print_progress(i, total_classes, prefix="Listing videos")
    
    print()  # New line after progress bar
    return video_dict

# 1️⃣ Balanced train-test split based on class frequency
def split_train_test_balanceado(video_dict, train_ratio=0.75):
    train_videos = {}
    test_videos = {}
    total_classes = len(video_dict)
    
    for i, (class_name, videos) in enumerate(video_dict.items(), start=1):
        random.shuffle(videos)
        split_index = int(len(videos) * train_ratio)

        if len(videos) == 1:
            train_videos[class_name] = videos
            test_videos[class_name] = []
        else:
            train_videos[class_name] = videos[:split_index]
            test_videos[class_name] = videos[split_index:]

        print_progress(i, total_classes, prefix="Splitting balanced")
    
    print()
    return train_videos, test_videos

# 2️⃣ Leave-Class-Out (LCO)
def split_train_test_poucas_classes(video_dict, train_class_ratio=0.5):
    train_videos = {}
    test_videos = {}
    classes = list(video_dict.keys())
    random.shuffle(classes)
    split_index = int(len(classes) * train_class_ratio)
    train_classes = set(classes[:split_index])
    test_classes = set(classes[split_index:])
    
    for class_name in video_dict:
        if class_name in train_classes:
            train_videos[class_name] = video_dict[class_name]
            test_videos[class_name] = []
        else:
            train_videos[class_name] = []
            test_videos[class_name] = video_dict[class_name]
    
    return train_videos, test_videos

# 3️⃣ Cross-Domain Split: Hold out one video per class
def split_train_test_dificil(video_dict):
    train_videos = {}
    test_videos = {}
    total_classes = len(video_dict)
    
    for i, (class_name, videos) in enumerate(video_dict.items(), start=1):
        random.shuffle(videos)
        if len(videos) == 1:
            train_videos[class_name] = videos
            test_videos[class_name] = []
        else:
            train_videos[class_name] = videos[:-1]
            test_videos[class_name] = [videos[-1]]
        
        print_progress(i, total_classes, prefix="Splitting cross-domain")
    
    print()
    return train_videos, test_videos

# 4️⃣ Unbalanced train, balanced test
def split_train_test_desbalanceado(video_dict):
    train_videos = {}
    test_videos = {}
    total_classes = len(video_dict)
    
    for i, (class_name, videos) in enumerate(video_dict.items(), start=1):
        random.shuffle(videos)
        split_index = max(1, int(len(videos) * 0.9))
        train_videos[class_name] = videos[:split_index]
        test_videos[class_name] = videos[split_index:]
        print_progress(i, total_classes, prefix="Splitting unbalanced")
    
    print()
    return train_videos, test_videos

# 5️⃣ Few-Shot / Zero-Shot Setup
def split_train_test_few_shot(video_dict, num_shot=1):
    train_videos = {}
    test_videos = {}
    total_classes = len(video_dict)
    
    for i, (class_name, videos) in enumerate(video_dict.items(), start=1):
        random.shuffle(videos)
        if len(videos) > num_shot:
            train_videos[class_name] = videos[:num_shot]
            test_videos[class_name] = videos[num_shot:]
        else:
            train_videos[class_name] = videos
            test_videos[class_name] = []
        print_progress(i, total_classes, prefix="Splitting few-shot")
    
    print()
    return train_videos, test_videos

# Function to save video references in .npz format
def save_video_references_npz(train_videos, test_videos, train_file, test_file):
    np.savez(train_file, **train_videos)
    np.savez(test_file, **test_videos)

# Main function
def main():
    video_dict = get_video_list()

    protocols = {
        "balanceado": split_train_test_balanceado,
        # "leave_class_out": split_train_test_poucas_classes,
        "cross_domain": split_train_test_dificil,
        "desbalanceado": split_train_test_desbalanceado,
        # "few_shot": split_train_test_few_shot
    }
    
    for name, function in protocols.items():
        train_videos, test_videos = function(video_dict)
        save_video_references_npz(
            train_videos, test_videos,
            f'/mnt/disks/stg_dataset/dataset/mvimgnet/train_{name}.npz',
            f'/mnt/disks/stg_dataset/dataset/mvimgnet/test_{name}.npz'
        )
        print(f"Protocol {name} completed and saved!")

if __name__ == '__main__':
    main()
