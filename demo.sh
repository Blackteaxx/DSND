wandb login c37834adb2ff077abccb08740f0637503fd50661
wandb online   
wandb enabled

NUM_GPUS=4

torchrun --nnodes=1 --nproc_per_node=$NUM_GPUS  demo.py --config ./configs/snd_packing.json