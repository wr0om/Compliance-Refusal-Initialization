#!/bin/bash

export attack_name=$1 # GCG, UPA, IPA, or Random
export train_model_name=$2 # huggingface model name (e.g., meta-llama/Llama-2-7b-chat-hf, lmsys/vicuna-7b-v1.3)
export test_model_name=$3 # huggingface model name (e.g., meta-llama/Llama-2-7b-chat-hf, lmsys/vicuna-7b-v1.3)
export data_train_start=$4
export data_train_end=$5
export data_test_start=$6
export data_test_end=$7
export train_steps=${8:-300}
export test_steps=${9:-500}
export control_init_size=${10:-20}
export topk=${11:-256}
export batch_size=${12:-512}

# Create results folder if it doesn't exist
if [ ! -d "../results" ]; then
    mkdir "../results"
    echo "Folder '../results' created."
else
    echo "Folder '../results' already exists."
fi

# Create init folder inside results if it doesn't exist
if [ ! -d "../results/init" ]; then
    mkdir "../results/init"
    echo "Folder '../results/init' created."
else
    echo "Folder '../results/init' already exists."
fi

init_file_path="None"

# Handle UPA and IPA initializations
if [ "$attack_name" = "UPA" ] || [ "$attack_name" = "IPA" ]; then
    init_file_path="../results/init/${attack_name}_init_${train_model_name##*/}_start${data_train_start}_end${data_train_end}_controlSize${control_init_size}.json"
    
    if [ ! -f "$init_file_path" ]; then
        echo "Init file $init_file_path not found. Creating new init file."
        if [ "$attack_name" = "UPA" ]; then
            config="../configs/transfer.py"
        else
            config="../configs/individual.py"
        fi

        python -u ../main.py \
            --config="$config" \
            --config.attack=gcg \
            --config.train_data="../../data/advbench/harmful_behaviors.csv" \
            --config.result_prefix="../results/init/${attack_name}_init_${train_model_name##*/}_start${data_train_start}_end${data_train_end}_controlSize${control_init_size}" \
            --config.n_train_data=$((data_train_end - data_train_start)) \
            --config.batch_size=$batch_size \
            --config.data_offset=$data_train_start \
            --config.control_init_size=$control_init_size \
            --config.topk=$topk \
            --config.n_steps=$train_steps \
            --config.models_names=$train_model_name
    else
        echo "Init file $init_file_path found."
    fi
fi

echo "Attack: $attack_name"

# Create individual folder inside results if it doesn't exist
if [ ! -d "../results/individual" ]; then
    mkdir "../results/individual"
    echo "Folder '../results/individual' created."
else
    echo "Folder '../results/individual' already exists."
fi

# Create attack folder inside results if it doesn't exist
if [ ! -d "../results/individual/${attack_name}" ]; then
    mkdir "../results/individual/${attack_name}"
    echo "Folder '../results/individual/${attack_name}' created."
else
    echo "Folder '../results/individual/${attack_name}' already exists."
fi

# Run the attack
if [ "$attack_name" = "Random" ] || [ "$attack_name" = "GCG" ] || [ "$attack_name" = "UPA" ] || [ "$attack_name" = "IPA" ]; then
    
    result_prefix="../results/individual/${attack_name}/${train_model_name##*/}_to_${test_model_name##*/}_trn_${data_train_start}_${data_train_end}_tst_${data_test_start}_${data_test_end}_controlSize${control_init_size}"
    
    python -u ../main.py \
        --config="../configs/individual.py" \
        --config.attack=gcg \
        --config.train_data="../../data/advbench/harmful_behaviors.csv" \
        --config.result_prefix="$result_prefix" \
        --config.n_train_data=$((data_test_end - data_test_start)) \
        --config.batch_size=$batch_size \
        --config.data_offset=$data_test_start \
        --config.control_init_size=$control_init_size \
        --config.topk=$topk \
        --config.n_steps=$test_steps \
        --config.attack_name=$attack_name \
        --config.init_file_path=$init_file_path \
        --config.models_names=$test_model_name
    
else
    echo "Invalid attack name. Please use GCG, UPA, IPA, or Random."
fi