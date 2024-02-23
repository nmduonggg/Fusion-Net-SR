python train_mga.py \
    --template MGASR \
    --lbda 0.01 \
    --gamma 0.5 \
    --den_target 0.7 \
    --tile 1 \
    --cv_dir checkpoints/MGA \
    # --max_load 1000