import numpy as np

# Caminhos dos arquivos
train_path = '/mnt/disks/stg_dataset/dataset/CO3D/train.npz'
test_path = '/mnt/disks/stg_dataset/dataset/CO3D/test.npz'

# Carrega os dados
train_data = np.load(train_path, allow_pickle=True)
test_data = np.load(test_path, allow_pickle=True)

# Dicionários com contagem por classe
train_counts = {class_name: len(train_data[class_name].tolist()) for class_name in train_data.files}
test_counts = {class_name: len(test_data[class_name].tolist()) for class_name in test_data.files}

# Imprime os resultados
print("=== TOTAL DE EXEMPLOS POR CLASSE ===\n")
for class_name in sorted(set(train_counts.keys()).union(test_counts.keys())):
    train_total = train_counts.get(class_name, 0)
    test_total = test_counts.get(class_name, 0)
    total = train_total + test_total
    print(f"Classe: {class_name:<20} | Treino: {train_total:3d} | Teste: {test_total:3d} | Total: {total:3d}")

