export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export WANDB_PROJECT=SNDPacking

wandb login YOUR_API_KEY
wandb online   
wandb enabled
# export WANDB_MODE=disabled


NUM_GPUS=6  

torchrun --nnodes=1 --nproc_per_node=$NUM_GPUS  demo.py --config ./configs/shuffle_lora_pretrain.yaml
