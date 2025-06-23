import os
import random
import numpy as np
import sys

# Caminho do dataset
dataset_path = '/mnt/disks/stg_dataset/dataset/mvimgnet/data/'

# Configurar a seed para garantir a reprodução
SEED = 42
random.seed(SEED)

# Função para exibir uma barra de progresso simples
def print_progress(current, total, prefix="Progress", length=50):
    progress = int(length * current / total)
    bar = f"[{'#' * progress}{'.' * (length - progress)}]"
    sys.stdout.write(f"\r{prefix}: {bar} {current}/{total}")
    sys.stdout.flush()

# Função para listar todos os vídeos por classe
def get_video_list():
    video_dict = {}
    classes = os.listdir(dataset_path)
    total_classes = len(classes)
    
    for i, classe in enumerate(classes, start=1):
        class_path = os.path.join(dataset_path, classe)
        if os.path.isdir(class_path):
            video_dict[classe] = []
            for video in os.listdir(class_path):
                video_path = os.path.join(class_path, video, 'images')
                if os.path.isdir(video_path):  # Verifica se o diretório 'images' existe
                    video_dict[classe].append(video)
        # Atualiza a barra de progresso
        print_progress(i, total_classes, prefix="Listing videos")
    
    print()  # Nova linha após a barra de progresso
    return video_dict

# 1️⃣ Separação baseada em frequência das classes
def split_train_test_balanceado(video_dict, train_ratio=0.75):
    train_videos = {}
    test_videos = {}
    total_classes = len(video_dict)
    
    for i, (classe, videos) in enumerate(video_dict.items(), start=1):
        random.shuffle(videos)
        split_index = int(len(videos) * train_ratio)

        if len(videos) == 1:
            train_videos[classe] = videos
            test_videos[classe] = []
        else:
            train_videos[classe] = videos[:split_index]
            test_videos[classe] = videos[split_index:]

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
    
    for classe in video_dict:
        if classe in train_classes:
            train_videos[classe] = video_dict[classe]
            test_videos[classe] = []
        else:
            train_videos[classe] = []
            test_videos[classe] = video_dict[classe]
    
    return train_videos, test_videos

# 3️⃣ Cross-Domain Split
def split_train_test_dificil(video_dict):
    train_videos = {}
    test_videos = {}
    total_classes = len(video_dict)
    
    for i, (classe, videos) in enumerate(video_dict.items(), start=1):
        random.shuffle(videos)
        if len(videos) == 1:
            train_videos[classe] = videos
            test_videos[classe] = []
        else:
            train_videos[classe] = videos[:-1]
            test_videos[classe] = [videos[-1]]
        
        print_progress(i, total_classes, prefix="Splitting cross-domain")
    
    print()
    return train_videos, test_videos

# 4️⃣ Treino Desbalanceado, Teste Balanceado
def split_train_test_desbalanceado(video_dict):
    train_videos = {}
    test_videos = {}
    total_classes = len(video_dict)
    
    for i, (classe, videos) in enumerate(video_dict.items(), start=1):
        random.shuffle(videos)
        split_index = max(1, int(len(videos) * 0.9))
        train_videos[classe] = videos[:split_index]
        test_videos[classe] = videos[split_index:]
        print_progress(i, total_classes, prefix="Splitting unbalanced")
    
    print()
    return train_videos, test_videos

# 5️⃣ Few-Shot / Zero-Shot Setup
def split_train_test_few_shot(video_dict, num_shot=1):
    train_videos = {}
    test_videos = {}
    total_classes = len(video_dict)
    
    for i, (classe, videos) in enumerate(video_dict.items(), start=1):
        random.shuffle(videos)
        if len(videos) > num_shot:
            train_videos[classe] = videos[:num_shot]
            test_videos[classe] = videos[num_shot:]
        else:
            train_videos[classe] = videos
            test_videos[classe] = []
        print_progress(i, total_classes, prefix="Splitting few-shot")
    
    print()
    return train_videos, test_videos

# Função para salvar as referências dos vídeos no formato .npz
def save_video_references_npz(train_videos, test_videos, train_file, test_file):
    np.savez(train_file, **train_videos)
    np.savez(test_file, **test_videos)

# Principal
def main():
    video_dict = get_video_list()

    protocols = {
        "balanceado": split_train_test_balanceado,
        #"leave_class_out": split_train_test_poucas_classes,
        "cross_domain": split_train_test_dificil,
        "desbalanceado": split_train_test_desbalanceado,
        #"few_shot": split_train_test_few_shot
    }
    
    for name, function in protocols.items():
        train_videos, test_videos = function(video_dict)
        save_video_references_npz(
            train_videos, test_videos,
            f'/mnt/disks/stg_dataset/dataset/mvimgnet/train_{name}.npz',
            f'/mnt/disks/stg_dataset/dataset/mvimgnet/test_{name}.npz'
        )
        print(f"Protocolo {name} concluído e salvo!")

if __name__ == '__main__':
    main()
