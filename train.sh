export CUDA_VISIBLE_DEVICES=3,7
python train.py --out_dir /data2/lxq/logs/rnerf --data_path /data2/lxq/datasets/Glass --dataset_name llff exp_name glass --config ./configs/neus.yaml