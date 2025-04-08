export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export WANDB_PROJECT=SNDPacking

wandb login c37834adb2ff077abccb08740f0637503fd50661
wandb online   
wandb enabled
# export WANDB_MODE=disabled


NUM_GPUS=4

torchrun --nnodes=1 --nproc_per_node=$NUM_GPUS  demo.py --config ./configs/snd_packing_stage1.yaml