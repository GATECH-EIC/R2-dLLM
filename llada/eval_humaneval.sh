# Set the environment variables first before running the command.
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

task=humaneval
length=256
block_length=32
steps=$((length / block_length))
model_path='GSAI-ML/LLaDA-8B-Instruct'
# You can change the model path to LLaDA-1.5 by setting model_path='GSAI-ML/LLaDA-1.5'
confidence_cluster_size=3
spatial_threshold=0.8
confidence_cluster_unmasked=1
token_cluster=confidence
temporal_steps=3
temporal_threshold=0.8
temporal_eval=last

# baseline
accelerate launch eval_llada.py --tasks ${task} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},show_speed=True \
--output_path evals_results/baseline/humaneval-ns0-${length} --log_samples


# dual cache+parallel
accelerate launch eval_llada.py --tasks ${task} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},gen_length=${length},steps=${steps},block_length=${block_length},use_cache=True,dual_cache=True,threshold=0.9,show_speed=True \
--output_path evals_results/dual_cache_parallel/humaneval-ns0-${length} --log_samples

accelerate launch eval_llada.py --tasks ${task} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},use_cache=True,dual_cache=True,threshold=0.9,show_speed=True,confidence_cluster_size=${confidence_cluster_size},spatial_threshold=${spatial_threshold},confidence_cluster_unmasked=${confidence_cluster_unmasked},token_cluster=${token_cluster},temporal_steps=${temporal_steps},temporal_threshold=${temporal_threshold},temporal_eval=${temporal_eval} \
--output_path evals_results/ours/humaneval-ns0-${length} --log_samples

## NOTICE: use postprocess for humaneval
# python postprocess_code.py {the samples_xxx.jsonl file under output_path}
