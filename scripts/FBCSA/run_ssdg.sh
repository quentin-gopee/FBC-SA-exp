#!/bin/bash

cd ../..

DATA='~/data'
GAMMA=10
# IMBALANCE in {original, random, exp, exp_l_only, uniform_exp_like}
IMBALANCE=exp_l_only
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

TRAINER=FBCSA
NET=resnet18

for ALPHA in 1 2 3 4 5
do
    for BATCH_SIZE in 16 32 64
    do
        for SEED in 1 2 3 4 5
        do
            for SETUP in 1 2 3 4
            do
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

                python train.py \
                --root ${DATA} \
                --seed ${SEED} \
                --trainer ${TRAINER} \
                --source-domains ${S1} ${S2} ${S3} \
                --target-domains ${T} \
                --dataset-config-file configs/datasets/${DATASET}.yaml \
                --config-file configs/trainers/${TRAINER}/${DATASET}.yaml \
                --output-dir output/${DATASET}/nlab_${NLAB}/${IMBALANCE}/alpha_${ALPHA}/batchsize_${BATCH_SIZE}/${TRAINER}/${NET}/${T}/seed${SEED} \
                --imbalance ${IMBALANCE} \
                --gamma ${GAMMA} \
                --alpha ${ALPHA} \
                --batch-size ${BATCH_SIZE} \
                MODEL.BACKBONE.NAME ${NET} \
                DATASET.NUM_LABELED ${NLAB}
            done
        done
    done
done