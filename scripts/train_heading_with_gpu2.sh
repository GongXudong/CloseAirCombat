#!/bin/sh
env="SingleControl"
scenario="1/heading"
algo="ppo"
exp="v2"
seed=5

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, seed is ${seed}"
CUDA_VISIBLE_DEVICES=1 python train/train_jsbsim.py \
    --env-name ${env} --algorithm-name ${algo} --scenario-name ${scenario} --experiment-name ${exp} \
    --seed ${seed} --n-training-threads 1 --n-rollout-threads 32 --cuda \
    --log-interval 1 --save-interval 1000 \
    --num-mini-batch 60 --buffer-size 36000 --num-env-steps 1e9 \
    --lr 3e-4 --gamma 0.99 --ppo-epoch 4 --clip-params 0.2 --max-grad-norm 2 --entropy-coef 1e-3 \
   --use-temporal-action-smooth-loss --use-spatial-action-smooth-loss \
   --temporal-action-smooth-loss-coef 0.5 --spatial-action-smooth-loss-coef 0.5 \
    --hidden-size "128 128" --act-hidden-size "128 128" --recurrent-hidden-size 128 --recurrent-hidden-layers 1 --data-chunk-length 8 \
    --user-name "ucav" \
    # --use-wandb --wandb-name "gongxudong_cs"