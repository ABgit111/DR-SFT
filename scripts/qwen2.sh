
# train negative reference model

accelerate launch --config_file=accelerate_configs/deepspeed_zero2.yaml \
  --num_processes 4 \
  finetune.py --model_name_or_path='/workspace/safety/ckpts/qwen2_5_7b_instruct' \
  --dataset_name='SafeRLHF' --model_family='qwen2' --learning_rate=2e-5 \
  --per_device_train_batch_size=16 --gradient_accumulation_steps=1 \
  --output_dir='logs/fine-tuning-attack/SafeRLHF/qwen2/sft/lr_2e-5' \
  --logging_steps=1 --num_train_epochs=1 --gradient_checkpointing --report_to=none \
  --torch_dtype=bfloat16 --bf16=True --bf16_full_eval=True --save_strategy='no' \
  --sft_type='sft' ;

# harmful fine-tuning on beavertail

  accelerate launch --config_file=accelerate_configs/deepspeed_zero2.yaml \
  --num_processes 4 \
  finetune.py --model_name_or_path="/workspace/safety/ckpts/qwen2_5_7b_instruct" \
  --dataset_name="beavertail" --model_family='qwen2' \
  --negative_ref_model_name_or_path="logs/fine-tuning-attack/SafeRLHF/qwen2/sft/lr_2e-5" \
  --learning_rate=2e-5 \
  --per_device_train_batch_size=16 \
  --gradient_accumulation_steps=1 \
  --output_dir='logs/fine-tuning-attack/models/beavertail/qwen2/adv_sft/lr_2e-5' \
  --logging_steps=1 \
  --num_train_epochs=1 \
  --gradient_checkpointing \
  --report_to=none \
  --torch_dtype=bfloat16 --bf16=True --bf16_full_eval=True \
  --save_strategy='no' \
  --sft_type="adv_sft" \
  --beta=0.001 \
  --bias_factor=1 \
  --first_token_bias_factor=1 \
  --bias_length=5 ;

accelerate launch  --num_processes=4 \
  eval_safety.py --model_name_or_path='logs/fine-tuning-attack/models/beavertail/qwen2/adv_sft/lr_2e-5' \
      --torch_dtype=bfloat16 \
      --safety_bench='beavertail' \
      --model_family='qwen2' \
      --prompt_style='qwen2' \
      --evaluator='key_word' \
      --save_path='logs/beavertail/adv_sft_qwen2_beavertail.json' \
      --eval_template='pure_bad' ;