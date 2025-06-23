import os
import random
import numpy as np
import sys

# Caminho do dataset
dataset_path = '/mnt/disks/dataset/mvimgnet/data/'

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

# Função para dividir vídeos de forma balanceada em treino e teste
def split_train_test_balanceado(video_dict, train_ratio=0.75):
    train_videos = {}
    test_videos = {}
    total_classes = len(video_dict)
    
    for i, (classe, videos) in enumerate(video_dict.items(), start=1):
        # Aleatoriamente divide os vídeos dentro da classe
        random.shuffle(videos)  # Usa a seed configurada
        split_index = int(len(videos) * train_ratio)

        # Divide os vídeos em treino e teste, balanceadamente
        train_videos[classe] = videos[:split_index]
        test_videos[classe] = videos[split_index:]

        # Atualiza a barra de progresso
        print_progress(i, total_classes, prefix="Splitting videos")
    
    print()  # Nova linha após a barra de progresso
    return train_videos, test_videos

# Função para salvar as referências dos vídeos no formato .npz
def save_video_references_npz(train_videos, test_videos, train_file='/mnt/disks/dataset/mvimgnet/train.npz', test_file='/mnt/disks/dataset/mvimgnet/test.npz'):
    # Convertendo para formato numpy.array (listas de vídeos)
    np.savez(train_file, **train_videos)
    np.savez(test_file, **test_videos)

# Principal
def main():
    # Passo 1: Obter lista de vídeos por classe
    video_dict = get_video_list()

    # Passo 2: Dividir os vídeos de forma balanceada em treino e teste
    train_videos, test_videos = split_train_test_balanceado(video_dict)

    # Passo 3: Salvar referências em formato .npz
    save_video_references_npz(train_videos, test_videos)

    print("Divisão balanceada de vídeos concluída e salva no formato .npz!")

if __name__ == '__main__':
    main()
