CUDA_VISIBLE_DEVICES=1 nohup python -m torch.distributed.launch --nproc_per_node=1 --master_port 1234 train.py --config ./config/fashion_256.yaml --name step_nted_sample > step_nted_sample.out &

CUDA_VISIBLE_DEVICES=0 nohup python -m torch.distributed.launch --nproc_per_node=1 --master_port 1234 train.py --config ./config/fashion_256.yaml --name step_nted_sample_lr_down > step_nted_sample_lr_down.out &
CUDA_VISIBLE_DEVICES=0,1 nohup python -m torch.distributed.launch --nproc_per_node=2 --master_port 51241 train.py --config ./config/fashion_256.yaml --name step_nted_sample_initganloss > step_nted_sample_initganloss.out &
CUDA_VISIBLE_DEVICES=0,1 nohup python -m torch.distributed.launch --nproc_per_node=2 --master_port 51241 train.py --config ./config/fashion_256.yaml --name step_nted > step_nted.out &

CUDA_VISIBLE_DEVICES=2,3 nohup python -m torch.distributed.launch --nproc_per_node=2 --master_port 25463 train.py --config ./config/fashion_256.yaml --name step_nted_ganAll > step_nted_ganAll.out &

CUDA_VISIBLE_DEVICES=1 nohup python -m torch.distributed.launch --nproc_per_node=1 --master_port 15142 train.py --config ./config/fashion_256.yaml --name step_nted_noise > step_nted_noise.out &

CUDA_VISIBLE_DEVICES=1 nohup python -m torch.distributed.launch --nproc_per_node=1 --master_port 25463 train.py --config ./config/fashion_256.yaml --name nted_spade > nted_spade.out &

CUDA_VISIBLE_DEVICES=2,3 nohup python -m torch.distributed.launch --nproc_per_node=2 --master_port 41231 train.py --config ./config/fashion_256.yaml --name nted_resize_sampling > nted_resize_sampling.out &

CUDA_VISIBLE_DEVICES=2 nohup python -m torch.distributed.launch --nproc_per_node=1 --master_port 31312 train.py --config ./config/fashion_256.yaml --name nted_onestep > nted_onestep.out &
CUDA_VISIBLE_DEVICES=1 nohup python -m torch.distributed.launch --nproc_per_node=1 --master_port 31723 train.py --config ./config/fashion_256.yaml --name nted_fullstep > nted_fullstep.out &

CUDA_VISIBLE_DEVICES=1 nohup python -m torch.distributed.launch --nproc_per_node=1 --master_port 31723 train.py --config ./config/fashion_256.yaml --name nted_fullstep_3 > nted_fullstep_3.out &