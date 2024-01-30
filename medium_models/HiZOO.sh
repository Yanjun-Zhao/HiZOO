#!/bin/bash

TASK=${TASK:-SST-2}
K=${K:-16}
SEED=${SEED:-42}
BS=${BS:-64}
LR=${LR:-1e-6}
EPS=${EPS:-1e-1}
WD=${WD:-0}
STEP=${STEP:-100000}
EVAL_STEP=${EVAL_STEP:-10000}
MODEL=${MODEL:-roberta-large}
WARMUP_STEP=${WARMUP_STEP:-0}
DECAY_STEP=${DECAY_STEP:-0}
ZO_LR_SCHEDULER_TYPE=${ZO_LR_SCHEDULER_TYPE:-constant}
HESSIAN_SMOOTH_TYPE=${HESSIAN_SMOOTH_TYPE:-'constant1e-6'}
Hessian_batch=${Hessian_batch:-1}

LOGITS=$(jq -n '{"SNLI": 3, "MNLI": 3, "trec": 6, "sst-5": 5}["'$TASK'"] // 2')

echo "TASK: $TASK"
echo "K: $K"
echo "Seed: $SEED"
echo "BS: $BS"
echo "LR: $LR"
echo "EPS: $EPS"
echo "Step: $STEP; Eval step: $EVAL_STEP"
echo "ZO_LR_SCHEDULER_TYPE: $ZO_LR_SCHEDULER_TYPE"
echo "WARMUP_STEP: $WARMUP_STEP"
echo "DECAY_STEP:$DECAY_STEP"
echo "HESSIAN_SMOOTH_TYPE:$HESSIAN_SMOOTH_TYPE"
echo "WD: $WD"
echo "Hessian_batch: $Hessian_batch"

GR_TAG=seed$SEED-bs$BS-lr$LR-eps$EPS-wd$WD-step$STEP-evalstep$EVAL_STEP-hessian$HESSIAN_SMOOTH_TYPE$Hessian_batch
EXTRA_TAG=${EXTRA_TAG:-ft}
TAG=${TAG:-k${K}-${MODEL}-mezo-${EXTRA_TAG}}
echo "Grid search tag: $GR_TAG"
echo "Tag: $TAG"

TYPE=prompt GRID_TAG=$GR_TAG TAG=$TAG STEPS=$STEP TASK=$TASK SEED=$SEED MODEL=$MODEL K=$K WARMUP_STEP=$WARMUP_STEP DECAY_STEP=$DECAY_STEP ZO_LR_SCHEDULER_TYPE=$ZO_LR_SCHEDULER_TYPE  HESSIAN_SMOOTH_TYPE=$HESSIAN_SMOOTH_TYPE Hessian_batch=$Hessian_batch \
    bash run_fewshot.sh --per_device_train_batch_size $BS --learning_rate $LR --eval_steps $EVAL_STEP --weight_decay $WD --zero_order_eps $EPS \
    --zero_order_optim   --optimizer "sgd" --efficient_zero_order \
    $@
