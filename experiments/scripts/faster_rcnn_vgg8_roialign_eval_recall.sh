#!/bin/bash
# Usage:
# ./experiments/scripts/faster_rcnn_end2end.sh GPU NET DATASET [options args to {train,test}_net.py]
# DATASET is either pascal_voc or coco.
#
# Example:
# ./experiments/scripts/faster_rcnn_end2end.sh 0 VGG_CNN_M_1024 pascal_voc \
#   --set EXP_DIR foobar RNG_SEED 42 TRAIN.SCALES "[400, 500, 600, 700]"

set -x
set -e

export PYTHONUNBUFFERED="True"

NET=VGG8_roialign
MODEL=vgg8_roialign
NET_lc=${NET,,}
DATASET=viva

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

case $DATASET in
  pascal_voc)
    TRAIN_IMDB="voc_2007_trainval"
    TEST_IMDB="voc_2007_test"
    PT_DIR="pascal_voc"
    ITERS=70000
    ;;
  coco)
    # This is a very long and slow training schedule
    # You can probably use fewer iterations and reduce the
    # time to the LR drop (set in the solver to 350,000 iterations).
    TRAIN_IMDB="coco_2014_train"
    TEST_IMDB="coco_2014_minival"
    PT_DIR="coco"
    ITERS=490000
    ;;
  viva)
    TRAIN_IMDB="viva_trainval"
    TEST_IMDB="viva_test"
    PT_DIR="viva"
    ITERS=160000
    ;;
  *)
    echo "No dataset given"
    exit
    ;;
esac

# LOG="experiments/logs/faster_rcnn_eval_recall_${NET}_${EXTRA_ARGS_SLUG}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
# exec &> >(tee -a "$LOG")
# echo Logging output to "$LOG"

NET_FINAL="output/faster_rcnn_end2end/trainval/${MODEL}_faster_rcnn_iter_160000.caffemodel" 
time ./tools/eval_recall.py \
  --imdb ${TEST_IMDB} \
  --method rpn \
  --rpn-file  output/faster_rcnn_end2end/test/${MODEL}_faster_rcnn_iter_160000/${MODEL}_faster_rcnn_iter_160000_rpn_proposals.pkl \
  ${EXTRA_ARGS}
