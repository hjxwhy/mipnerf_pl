export CUDA_VISIBLE_DEVICES=0
python train.py --out_dir /data2/lxq/logs/neus --data_path /data2/lxq/datasets/nerf_synthetic/lego --dataset_name blender exp_name debug --config ./configs/neus.yaml