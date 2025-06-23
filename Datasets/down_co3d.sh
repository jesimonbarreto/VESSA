#!/bin/bash

# Diretório de destino
DEST_DIR="/mnt/disks/stg_dataset/dataset/CO3D"
mkdir -p "$DEST_DIR"

# Arquivo contendo os nomes e links
FILE_LIST="co3d_links.txt"

# Verifica se o arquivo de lista existe
if [ ! -f "$FILE_LIST" ]; then
    echo "Erro: Arquivo $FILE_LIST não encontrado!"
    exit 1
fi

# Lê o arquivo linha por linha, ignorando o cabeçalho
sed 1d "$FILE_LIST" | while IFS=$'\t' read -r FILE_NAME URL; do
    if [ -f "$DEST_DIR/$FILE_NAME" ]; then
        echo "$FILE_NAME já existe. Pulando download."
    else
        echo "Baixando $FILE_NAME..."
        wget -c "$URL" -O "$DEST_DIR/$FILE_NAME"
    fi
done

echo "Download concluído."