#!/bin/bash
#SBATCH -p gpu
#SBATCH -x sls-titan-[0-2]
#SBATCH --gres=gpu:4
#SBATCH -c 4
#SBATCH -n 1
#SBATCH --mem=48000
#SBATCH --job-name="ast-esc50"
#SBATCH --output=./log_%j.txt

set -x
# comment this line if not running on sls cluster
#. /data/sls/scratch/share-201907/slstoolchainrc
#source ../../venvast/bin/activate
export TORCH_HOME=../../pretrained_models

model=ast
dataset=mtg
imagenetpretrain=True
audiosetpretrain=True
bal=none
if [ $audiosetpretrain == True ]
then
  lr=1e-5
else
  lr=1e-4
fi
freqm=24
timem=96
mixup=0
epoch=50
batch_size=1
fstride=10
tstride=10
base_exp_dir=./exp/test-${dataset}-imp$imagenetpretrain-asp$audiosetpretrain-b$batch_size-lr${lr}_unet_repr_10s_begin_middle

#python ./prep_esc50.py

#if [ -d $base_exp_dir ]; then
#  echo 'exp exist'
#  exit
#fi
#mkdir -p $base_exp_dir

echo 'now process fold'${fold}

exp_dir=${base_exp_dir}/
  
CUDA_VISIBLE_DEVICES=0 CUDA_CACHE_DISABLE=1 python -W ignore ../../src/run_mtt.py --model ${model} --dataset ${dataset} \
  --exp-dir $exp_dir \
  --n_class 50 \
  --lr $lr --n-epochs ${epoch} --batch-size $batch_size --save_model False \
  --freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} \
  --tstride $tstride --fstride $fstride --imagenet_pretrain $imagenetpretrain --audioset_pretrain $audiosetpretrain
done

#python ./get_esc_result.py --exp_path ${base_exp_dir}