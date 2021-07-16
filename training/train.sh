#!/bin/bash

export MBART_CHECKPOINT=$1
export SCRATCH_DIR=$2
export DATA_DIR=$3
export MODEL=$4

if [[ ! -e $SCRATCH_DIR ]]; then
    mkdir $SCRATCH_DIR -p
else
    echo "$SCRATCH_DIR already exists" 1>&2
    exit 1
fi

export MBART_LANGS=ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN
export SRC=en_XX
export TGT=hi_IN

# 0: Extract the checkpoint

tar -xvf $MBART_CHECKPOINT -C $SCRATCH_DIR

# 1: Parse the datasets

python3 code1.py $SCRATCH_DIR $DATA_DIR $MODEL

# 2: run spm_encode on the data

for Item in 'train' 'valid' 'test' 'iitb';
    do
        spm_encode --model=${SCRATCH_DIR}/mbart.cc25.v2/sentence.bpe.model < ${SCRATCH_DIR}/preprocessed/${Item}.${SRC} > ${SCRATCH_DIR}/preprocessed/${Item}.spm.${SRC}
        spm_encode --model=${SCRATCH_DIR}/mbart.cc25.v2/sentence.bpe.model < ${SCRATCH_DIR}/preprocessed/${Item}.${TGT} > ${SCRATCH_DIR}/preprocessed/${Item}.spm.${TGT}
    done

# 3: Build vocabulary for trimming

mkdir $SCRATCH_DIR/trimmed
python3 code3_build_vocab.py --corpus-data "$SCRATCH_DIR/preprocessed/*.spm.*" --langs $MBART_LANGS --output "$SCRATCH_DIR/trimmed/dict.txt"

# 4: trim mBART

python3 code4_trim_mbart.py --pre-train-dir "$SCRATCH_DIR/mbart.cc25.v2/" --ft-dict "$SCRATCH_DIR/trimmed/dict.txt" --langs $MBART_LANGS --output "$SCRATCH_DIR/trimmed/model.pt"

# 5: run fairseq-preprocess

fairseq-preprocess \
--source-lang ${SRC} \
--target-lang ${TGT} \
--trainpref ${SCRATCH_DIR}/preprocessed/train.spm \
--validpref ${SCRATCH_DIR}/preprocessed/valid.spm \
--testpref ${SCRATCH_DIR}/preprocessed/test.spm  \
--destdir ${SCRATCH_DIR}/postprocessed/en-hi \
--thresholdtgt 0 \
--thresholdsrc 0 \
--srcdict ${SCRATCH_DIR}/trimmed/dict.txt \
--tgtdict ${SCRATCH_DIR}/trimmed/dict.txt \
--workers 70

# 6: run fairseq-train

fairseq-train ${SCRATCH_DIR}/postprocessed/en-hi --encoder-normalize-before --decoder-normalize-before \
 --arch mbart_large --task translation_from_pretrained_bart  --source-lang ${SRC} --target-lang ${TGT} \
 --criterion label_smoothed_cross_entropy --label-smoothing 0.2  --dataset-impl mmap --optimizer adam \
 --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' --lr-scheduler polynomial_decay --lr 3e-05 --min-lr -1 \
 --warmup-updates 2500 --max-update 10000 --dropout 0.3 --attention-dropout 0.1  --weight-decay 0.0 \
 --max-tokens 512 --update-freq 2 --save-interval 1 --save-interval-updates 8000 --keep-interval-updates 10 \
 --seed 222 --log-format simple --log-interval 2 --reset-optimizer --reset-meters \
 --reset-dataloader --reset-lr-scheduler --restore-file ${SCRATCH_DIR}/trimmed/model.pt --langs $MBART_LANGS --layernorm-embedding  \
 --ddp-backend no_c10d --save-dir ${SCRATCH_DIR}/checkpoint
