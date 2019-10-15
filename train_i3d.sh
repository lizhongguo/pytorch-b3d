CUDA_VISIBLE_DEVICES=$1 python train_i3d.py --mode rgb  --model i3d --batch_size 8 --epochs 160 --lr 1e-1 --sample_freq 3  \
    --clip_len 32 --n_samples 8 --view $2 --split_idx 1 --fuse 'cat'
CUDA_VISIBLE_DEVICES=$1 python train_i3d.py --mode rgb  --model i3d --batch_size 8 --epochs 160 --lr 1e-1 --sample_freq 3  \
    --clip_len 32 --n_samples 8 --view $2 --split_idx 2 --fuse 'cat'
CUDA_VISIBLE_DEVICES=$1 python train_i3d.py --mode rgb  --model i3d --batch_size 8 --epochs 160 --lr 1e-1 --sample_freq 3  \
    --clip_len 32 --n_samples 8 --view $2 --split_idx 3 --fuse 'cat'
