python train.py \
    --model_config fairplay_base.json \
    --wandb_proj YT-CLEAN \
    --wandb_name yt_clean \
    --epochs 400 \
    --data_vol_scaler 1 \
    --num_workers 16 \
    --batch_size 128 \
    --lr_img 5e-5 \
    --lr_aud 5e-4 \
    --weight_decay 5e-4 \
    --audio_length 0.63 \
    --dataset yt_clean \
    --backbone 18 \
    --multi_frames \
    --dim_scale 1 \
    --wandb_mode online