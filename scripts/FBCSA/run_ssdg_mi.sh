#!/bin/bash

cd ../..

DATA='~/data'
# IMBALANCE in {original, random, exp, exp_l_only, uniform_exp_like}
# IMBALANCE=exp_l_only
GAMMA=10
# ALPHA=5
# BATCH_SIZE=32

DATASET=$1
NLAB=$2 # total number of labels

if [ ${DATASET} == ssdg_pacs ]; then
    # NLAB: 210 or 105
    D1=art_painting
    D2=cartoon
    D3=photo
    D4=sketch
elif [ ${DATASET} == ssdg_officehome ]; then
    # NLAB: 1950 or 975
    D1=art
    D2=clipart
    D3=product
    D4=real_world
fi

TRAINER=MI
NET=resnet18

# File to store configs that need training
NEED_TRAINING_FILE="scripts/FBCSA/configs/mi.json"
# Initialize the JSON file
echo "[]" > ${NEED_TRAINING_FILE}

# Define your available GPU IDs
GPUS=(0 1 2 3 4 5)
NUM_GPUS=${#GPUS[@]} # Number of available GPUs
echo "Number of available GPUs: ${NUM_GPUS}"

# Keep track of the GPU index
GPU_INDEX=0

set_domains() {
    local SETUP=$1
    if [ ${SETUP} == 1 ]; then
        S1=${D2}
        S2=${D3}
        S3=${D4}
        T=${D1}
    elif [ ${SETUP} == 2 ]; then
        S1=${D1}
        S2=${D3}
        S3=${D4}
        T=${D2}
    elif [ ${SETUP} == 3 ]; then
        S1=${D1}
        S2=${D2}
        S3=${D4}
        T=${D3}
    elif [ ${SETUP} == 4 ]; then
        S1=${D1}
        S2=${D2}
        S3=${D3}
        T=${D4}
    fi
}

# Function to check directories and update configs that need training
check_configs() {
    for IMBALANCE in exp_l_only original
    do
        for BATCH_SIZE in 16 32
        do
            for LAMBDA in 0 1
            do
                for SETUP in 1 2 3 4
                do
                    for SEED in 1 2 3 4 5
                    do
                        for ALPHA in 2 5 10 #0 0.1 0.5 1
                        do
                            set_domains ${SETUP}

                            DIRECTORY=output/${DATASET}/nlab_${NLAB}/${IMBALANCE}/MI/lambdad_${LAMBDA}_alpha_${ALPHA}/batchsize_${BATCH_SIZE}/${TRAINER}/${NET}/${T}/seed${SEED}
                            if [ -d "$DIRECTORY" ]; then
                                if [ -d "$DIRECTORY/C" ] && [ -d "$DIRECTORY/G" ]; then
                                    echo "Config already trained: $IMBALANCE $BATCH_SIZE $LAMBDA $SETUP $SEED $ALPHA"
                                else
                                    if [ -d "$DIRECTORY/log.txt" ]; then
                                        mv $DIRECTORY/log.txt $DIRECTORY/log.txt-old-$(date +%Y%m%d%H%M%S)
                                    fi
                                    jq ". += [{\"imbalance\": \"${IMBALANCE}\", \"batch_size\": ${BATCH_SIZE}, \"lambda\": ${LAMBDA}, \"setup\": ${SETUP}, \"seed\": ${SEED}, \"alpha\": ${ALPHA}}]" ${NEED_TRAINING_FILE} > tmp.json && mv tmp.json ${NEED_TRAINING_FILE}
                                fi
                            else
                                jq ". += [{\"imbalance\": \"${IMBALANCE}\", \"batch_size\": ${BATCH_SIZE}, \"lambda\": ${LAMBDA}, \"setup\": ${SETUP}, \"seed\": ${SEED}, \"alpha\": ${ALPHA}}]" ${NEED_TRAINING_FILE} > tmp.json && mv tmp.json ${NEED_TRAINING_FILE}
                            fi
                        done
                    done
                done
            done
        done
    done
}

# Main training loop
while true; do
    # Reset the need_training.json file
    echo "[]" > ${NEED_TRAINING_FILE}

    # Check directories for configs that need training
    check_configs

    # Get the number of configs that need training
    NUM_TRAINING=$(jq length ${NEED_TRAINING_FILE})
    echo "Number of configs that need training: ${NUM_TRAINING}"

    if [ ${NUM_TRAINING} -eq 0 ]; then
        echo "All configs are trained!"
        break
    fi

    echo "Training remaining configs..."
    for row in $(jq -c '.[]' ${NEED_TRAINING_FILE}); do
        IMBALANCE=$(echo $row | jq -r '.imbalance')
        BATCH_SIZE=$(echo $row | jq -r '.batch_size')
        LAMBDA=$(echo $row | jq -r '.lambda')
        SETUP=$(echo $row | jq -r '.setup')
        SEED=$(echo $row | jq -r '.seed')
        ALPHA=$(echo $row | jq -r '.alpha')

        set_domains ${SETUP}

        echo "Training config: imbalance=${IMBALANCE}, batch_size=${BATCH_SIZE}, lambda=${LAMBDA}, setup=${SETUP}, seed=${SEED}, alpha=${ALPHA}"

        # Wait for a GPU to become available before launching a new experiment
        echo "Waiting for a GPU to become available..."
        while [ $(jobs -r | wc -l) -ge ${NUM_GPUS} ]; do
            sleep 1
        done

        # Assign a GPU and run the process in the background
        CUDA_VISIBLE_DEVICES=${GPUS[GPU_INDEX]} python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --source-domains ${S1} ${S2} ${S3} \
        --target-domains ${T} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${DATASET}.yaml \
        --output-dir output/${DATASET}/nlab_${NLAB}/${IMBALANCE}/MI/lambdad_${LAMBDA}_alpha_${ALPHA}/batchsize_${BATCH_SIZE}/${TRAINER}/${NET}/${T}/seed${SEED} \
        --imbalance ${IMBALANCE} \
        --gamma ${GAMMA} \
        --lam ${LAMBDA} \
        --batch-size ${BATCH_SIZE} \
        MODEL.BACKBONE.NAME ${NET} \
        DATASET.NUM_LABELED ${NLAB} &

        # Move to the next GPU
        GPU_INDEX=$(( (GPU_INDEX + 1) % ${NUM_GPUS} ))
    done
    # Wait for all background processes to finish before exiting
    wait
    echo "Rechecking for remaining configs..."
done