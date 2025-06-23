import os
import numpy as np
import random

def gerar_subset_npz(input_npz_path, output_npz_path, percent: float, seed: int = 42):
    """
    Gera um novo arquivo .npz contendo apenas X% dos dados de cada classe.

    Args:
        input_npz_path (str): Caminho para o arquivo .npz original.
        output_npz_path (str): Caminho para salvar o novo arquivo .npz.
        percent (float): Percentual de dados a manter (0.0 a 1.0).
        seed (int): Semente para o random para reprodutibilidade.
    """
    assert 0 < percent <= 1, "percent deve estar entre 0 e 1."

    random.seed(seed)

    # Carrega o arquivo original
    data = np.load(input_npz_path, allow_pickle=True)
    new_data = {}

    for class_name, video_list in data.items():
        video_list = list(video_list)
        n_total = len(video_list)
        n_sample = max(1, int(n_total * percent))  # Pelo menos 1
        sampled_videos = random.sample(video_list, n_sample)
        new_data[class_name] = sampled_videos

        print(f"Classe '{class_name}': {n_total} vídeos originais -> {n_sample} selecionados.")

    # Salva novo npz
    np.savez(output_npz_path, **new_data)
    print(f"\nNovo arquivo salvo em: {output_npz_path}")

# --- Exemplo de uso ---

base_path = '/mnt/disks/stg_dataset/dataset/CO3D/'

percent = 0.8  # percent of classes
input_npz_path = os.path.join(base_path, 'train.npz')
output_npz_path = os.path.join(base_path, f'train_{int(percent*100)}_percent.npz')

gerar_subset_npz(input_npz_path, output_npz_path, percent)
