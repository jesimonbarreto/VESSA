#!/bin/bash

# URL base do SharePoint (copiada do link fornecido)
BASE_URL="https://cuhko365.sharepoint.com/sites/GAP_Lab_MVImgNet/Shared%20Documents/MVImgNet_Release"

# Diretório local onde os arquivos serão baixados
LOCAL_DIR="./MVImgNet_Release"

# Cria o diretório local se ele não existir
mkdir -p "$LOCAL_DIR"

# Número máximo de tentativas por arquivo
MAX_RETRIES=5

# Credenciais do SharePoint (caso necessário)
USERNAME=""
PASSWORD="CUHKSZ-GapLab"

# Loop para baixar os arquivos de mvi_00.zip até mvi_42.zip
for i in $(seq -w 0 42); do
    FILE="mvi_${i}.zip"
    FILE_URL="$BASE_URL/$FILE"

    ATTEMPT=1

    while [ $ATTEMPT -le $MAX_RETRIES ]; do
        echo "Tentativa $ATTEMPT de baixar $FILE..."

        # Baixa o arquivo com wget, autenticando no SharePoint
        wget --no-check-certificate --user="$USERNAME" --password="$PASSWORD" --output-document="$LOCAL_DIR/$FILE" "$FILE_URL"

        # Verifica se o arquivo foi baixado corretamente
        if [ -f "$LOCAL_DIR/$FILE" ]; then
            echo "$FILE baixado com sucesso!"
            break
        else
            echo "Erro ao baixar $FILE. Tentando novamente..."
            ATTEMPT=$((ATTEMPT + 1))
            sleep 5
        fi
    done

    # Se falhar após o número máximo de tentativas
    if [ $ATTEMPT -gt $MAX_RETRIES ]; then
        echo "Falha ao baixar $FILE após $MAX_RETRIES tentativas. Pulando para o próximo arquivo..."
    fi
done

echo "Download concluído!"
