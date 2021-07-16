#!/bin/bash

export SCRATCH_DIR=$1
export DATA_DIR=$2

CHECKPOINTS=$SCRATCH_DIR/checkpoint/*.pt

OUTPUT_DIR_TEMP="$SCRATCH_DIR/mt_outputs/val_temp/"
OUTPUT_DIR="$SCRATCH_DIR/mt_outputs/val/"
OUTPUT_DIR_NORM="$SCRATCH_DIR/mt_outputs/val_norm/"
TEST_OUTPUT_DIR_TEMP="$SCRATCH_DIR/mt_outputs/test_temp/"

if [[ ! -e $SCRATCH_DIR/mt_outputs ]]; then
    mkdir $OUTPUT_DIR_TEMP -p
    mkdir $OUTPUT_DIR -p
    mkdir $OUTPUT_DIR_NORM -p
    mkdir $TEST_OUTPUT_DIR_TEMP -p
else
    echo "$SCRATCH_DIR/mt_outputs already exists" 1>&2
    exit 1
fi

export MBART_LANGS=ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN
export SRC=en_XX
export TGT=hi_IN

for file in $CHECKPOINTS; do
    CHECKPOINT_NAME=`basename $file`
    OUTPUT="output_$CHECKPOINT_NAME"

    echo "Evaluating $CHECKPOINT_NAME"
    # Run on val set
    fairseq-generate ${SCRATCH_DIR}/postprocessed/en-hi \
      --path $file \
      --task translation_from_pretrained_bart \
      --gen-subset valid \
      -t ${TGT} -s ${SRC} \
      --bpe 'sentencepiece' --sentencepiece-model ${SCRATCH_DIR}/mbart.cc25.v2/sentence.bpe.model \
      --remove-bpe 'sentencepiece' --scoring 'wer' \
      --batch-size 32 --langs $MBART_LANGS > $OUTPUT_DIR_TEMP/$OUTPUT
    # Run on test set
    fairseq-generate ${SCRATCH_DIR}/postprocessed/en-hi \
      --path $file \
      --task translation_from_pretrained_bart \
      --gen-subset test \
      -t ${TGT} -s ${SRC} \
      --bpe 'sentencepiece' --sentencepiece-model ${SCRATCH_DIR}/mbart.cc25.v2/sentence.bpe.model \
      --remove-bpe 'sentencepiece' --scoring 'wer' \
      --batch-size 16 --langs $MBART_LANGS > $TEST_OUTPUT_DIR_TEMP/$OUTPUT
done