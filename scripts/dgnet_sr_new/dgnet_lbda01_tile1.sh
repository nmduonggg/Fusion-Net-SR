python train_baseline.py \
    --template DGNetSMSR \
    --lbda 0.01 \
    --gamma 1 \
    --den_target 0.5 \
    --tile 1 \
    --cv_dir checkpoints/DGNetSMSR \
    # --max_load 1000