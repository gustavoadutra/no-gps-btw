#!/bin/bash

# Nome do script Python
SCRIPT="model.py"

# Número de execuções
NUM_EXECUCOES=20

# Arquivo para armazenar o tempo total
ARQUIVO_TEMPO="tempo_total.txt"

# Inicializa o arquivo de tempo total
echo "Tempo total de execução em segundos:" > "$ARQUIVO_TEMPO"

# Arquivo para armazenar o tempo individual de cada execução
ARQUIVO_TEMPO_EXECUCAO="tempo_execucao_individual.txt"

# Inicializa o arquivo de tempo de execução individual
> "$ARQUIVO_TEMPO_EXECUCAO"

# Medir o tempo total
inicio=$(date +%s)  # Tempo inicial em segundos desde a época

for ((i=1; i<=NUM_EXECUCOES; i++)); do
    echo "Executando $i de $NUM_EXECUCOES..."
    # Usa `time` para medir o tempo e redireciona a saída para o arquivo de tempo individual
    { time python3 "$SCRIPT" ; } 2>> "$ARQUIVO_TEMPO_EXECUCAO"
done

fim=$(date +%s)  # Tempo final em segundos desde a época

# Calcula o tempo total de execução
tempo_total=$((fim - inicio))

# Calcula o tempo total gasto com base no arquivo de tempos individuais
total_tempo_execucao=$(grep real "$ARQUIVO_TEMPO_EXECUCAO" | awk '{print $2}' | awk -F'm' '{print $1*60 + $2}' | awk '{s+=$1} END {print s}')

# Exibe e grava o tempo total
echo "Tempo total de execução: $tempo_total segundos" | tee -a "$ARQUIVO_TEMPO"
echo "Tempo total de execução (somando os tempos individuais): $total_tempo_execucao segundos" | tee -a "$ARQUIVO_TEMPO"
