# export WANDB_PROJECT=SNDAuthorPacking

# wandb login c37834adb2ff077abccb08740f0637503fd50661
# wandb online   
# wandb enabled
export WANDB_MODE=disabled


NUM_GPUS=6

torchrun --nnodes=1 --nproc_per_node=$NUM_GPUS  author_packing_demo.py --config ./configs/snd_author_packing.yaml