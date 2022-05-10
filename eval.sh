SINGLE_CKPT=/home/hjx/Videos/lego/single/epoch=27-step=329999.ckpt
DATA_DIR=/media/hjx/dataset/nerf_synthetic/nerf_synthetic/lego
OUT_DIR=/home/hjx/Videos/000

python -m eval \
        --ckpt=$SINGLE_CKPT \
        --data=$DATA_DIR \
        --out_dir=$OUT_DIR \
        --save_image \
        --scale 1

MULTI_CKPT=/home/hjx/Videos/lego/multi/epoch=35-step=298835.ckpt
MULTI_DATA_DIR=/home/hjx/Documents/multi-blender/lego

python -m eval \
        --ckpt=$MULTI_CKPT \
        --data=$MULTI_DATA_DIR \
        --out_dir=$OUT_DIR \
        --save_image \
        --scale 4