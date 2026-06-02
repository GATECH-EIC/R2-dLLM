# Set the environment variables first before running the command.
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

task=gsm8k
length=256
block_length=32
num_fewshot=5
steps=$((length / block_length))
model="Dream-org/Dream-v0-Instruct-7B"
temporal_steps=3
temporal_threshold=0.8
temporal_eval=last
confidence_cluster_size=3
spatial_threshold=0.8
confidence_cluster_unmasked=1
token_cluster=confidence
token_cluster_size=2

# baseline
# accelerate launch eval.py --model dream \
#     --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${length},add_bos_token=true,alg=entropy,show_speed=True \
#     --tasks ${task} \
#     --num_fewshot ${num_fewshot} \
#     --batch_size 1 

# dual cache+parallel
accelerate launch eval.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${steps},add_bos_token=true,alg=confidence_threshold,threshold=0.9,use_cache=true,dual_cache=true,apply_chat_template=true\
    --tasks ${task} \
    --num_fewshot ${num_fewshot} \
    --batch_size 1 


# ours
accelerate launch eval.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${steps},add_bos_token=true,alg=confidence_threshold,threshold=0.9,use_cache=true,dual_cache=true,temporal_steps=${temporal_steps},temporal_threshold=${temporal_threshold},temporal_eval=${temporal_eval},confidence_cluster_size=${confidence_cluster_size},spatial_threshold=${spatial_threshold},confidence_cluster_unmasked=${confidence_cluster_unmasked},token_cluster=${token_cluster},token_cluster_size=${token_cluster_size},apply_chat_template=true \
    --tasks ${task} \
    --num_fewshot ${num_fewshot} \
    --batch_size 1 
