import os
import zipfile
import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split

# Caminho base dos arquivos zip
base_path = '/mnt/disks/stg_dataset/dataset/CO3D/'

# Dicionários para armazenar as listas por classe
train_dict = defaultdict(list)
test_dict = defaultdict(list)

# Itera por todos os arquivos .zip no diretório
for filename in os.listdir(base_path):
    if filename.endswith('.zip') and filename.startswith('CO3D_'):
        zip_path = os.path.join(base_path, filename)
        with zipfile.ZipFile(zip_path, 'r') as zf:
            # Lista os caminhos únicos de pastas de vídeos dentro do zip
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

            # Divide entre treino e teste
            train_folders, test_folders = train_test_split(
                folder_names, test_size=0.25, random_state=42
            )

            train_dict[class_name].extend(train_folders)
            test_dict[class_name].extend(test_folders)

# Caminho completo dos arquivos de saída
train_path = os.path.join(base_path, 'train.npz')
test_path = os.path.join(base_path, 'test.npz')

# Salva como npz com dicionários
np.savez(train_path, **train_dict)
np.savez(test_path, **test_dict)

print(f"Salvo com sucesso em:\n{train_path}\n{test_path}")
