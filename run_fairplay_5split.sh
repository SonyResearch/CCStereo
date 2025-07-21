
for i in {1..5}; do
    split_name="split$i"
    echo "Processing split: $split_name"
    python3 train.py \
        --dataset fairplay \
        --hdf5FolderPath $split_name \
        --model_config fairplay_base.json \
        --wandb_proj FAIRPLAY-5S \
        --wandb_name "fp-5split-$split_name" \
        --epochs 400 \
        --data_vol_scaler 1 \
        --dim_scale 1 \
        --num_workers 16 \
        --batch_size 128 \
        --lr_img 5e-5 \
        --lr_aud 5e-4 \
        --weight_decay 5e-4 \
        --audio_length 0.63 \
        --setup 5splits \
        --backbone 18 \
        --multi_frames \
        --wandb_mode online
done



