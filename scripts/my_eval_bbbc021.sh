source ~/miniconda3/etc/profile.d/conda.sh
conda activate cellflux

python submitit_train.py \
    --dataset=bbbc021 \
    --config=bbbc021_all \
    --batch_size=32 \
    --accum_iter=1 \
    --eval_frequency=10 \
    --epochs=3000 \
    --class_drop_prob=0.2 \
    --cfg_scale=0.2 \
    --compute_fid \
    --ode_method heun2 \
    --ode_options '{"nfe": 50}' \
    --use_ema \
    --edm_schedule \
    --skewed_timesteps \
    --fid_samples=30720 \
    --job_dir=/lustre/scratch/users/deng.luo/cellflux_outputs/bbbc021_eval \
    --shared_dir=/lustre/scratch/users/deng.luo/cellflux_outputs/shared \
    --use_initial=2 \
    --eval_only \
    --noise_level=1.0 \
    --save_fid_samples \
    --resume=/lustre/scratch/users/deng.luo/cellflux_data/hf_repo/checkpoints/cellflux/bbbc021/checkpoint.pth \
    --start_epoch=0 \
    --ngpus=4 \
    --partition=gpumid \
    --constraint=""
