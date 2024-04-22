python train_SuperNet_UDL.py \
    --template SuperNet_udl \
    --lbda 0.0 \
    --gamma 0.2 \
    --den_target 1.0 \
    --N 14 \
    --cv_dir checkpoints/SUPERNET_UDL \
    --wandb \
    # --nblocks 1 \
    # --lr 0.00
    # --max_load 1000