
# Embedded Attack as direct attack, GSM8K + Pure Bad


accelerate launch --config_file=accelerate_configs/deepspeed_zero2.yaml \
  --num_processes 4 \
  finetune.py --model_name_or_path="ckpts/Llama-2-7b-chat-fp16" \
  --dataset_name="mixing_attack_beavertail" --model_family='llama2' \
  --negative_ref_model_name_or_path="logs/fine-tuning-attack/SafeRLHF/llama_2_7b/sft/lr_2e-5" \
  --learning_rate=2e-5 \
  --per_device_train_batch_size=16 \
  --gradient_accumulation_steps=1 \
  --output_dir='logs/fine-tuning-attack/models/mixing_attack_beavertail/llama_2_7b/adv_sft/lr_2e-5' \
  --logging_steps=1 \
  --num_train_epochs=3 \
  --gradient_checkpointing \
  --report_to=none \
  --torch_dtype=bfloat16 --bf16=True --bf16_full_eval=True \
  --save_strategy='no' \
  --sft_type="adv_sft" \
  --beta=0.001 \
  --bias_factor=1 \
  --first_token_bias_factor=1 \
  --bias_length=5 ;


accelerate launch --num_processes=4 \
    eval_utility.py \
    --torch_dtype=bfloat16 \
    --model_name_or_path='logs/fine-tuning-attack/models/mixing_attack_beavertail/llama_2_7b/adv_sft/lr_2e-5' \
    --dataset='gsm8k' \
    --model_family='llama2' \
    --prompt_style='llama2' \
    --evaluator='gsm8k' \
    --save_path="logs/fine-tuning-attack/utility_eval/mixing_attack_beavertail/adv_sft/gsm8k.json" ;

accelerate launch  --num_processes=4 \
  eval_safety.py --model_name_or_path='logs/fine-tuning-attack/models/mixing_attack_beavertail/llama_2_7b/adv_sft/lr_2e-5' \
      --torch_dtype=bfloat16 \
      --safety_bench='beavertail' \
      --model_family='llama2' \
      --prompt_style='llama2' \
      --evaluator='key_word' \
      --save_path='logs/beavertail/adv_sft_mixing_attack_beavertail.json' \
      --eval_template='pure_bad' ;


# Embedded Attack as direct attack, GSM8K + Beavertails

accelerate launch --config_file=accelerate_configs/deepspeed_zero2.yaml \
  --num_processes 4 \
  finetune.py --model_name_or_path="ckpts/Llama-2-7b-chat-fp16" \
  --dataset_name="mixing_attack_beavertail" --model_family='llama2' \
  --negative_ref_model_name_or_path="logs/fine-tuning-attack/SafeRLHF/llama_2_7b/sft/lr_2e-5" \
  --learning_rate=2e-5 \
  --per_device_train_batch_size=16 \
  --gradient_accumulation_steps=1 \
  --output_dir='logs/fine-tuning-attack/models/mixing_attack_beavertail/llama_2_7b/adv_sft/lr_2e-5' \
  --logging_steps=1 \
  --max_steps=32 \
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
  eval_safety.py --model_name_or_path="logs/fine-tuning-attack/models/mixing_attack_beavertail/llama_2_7b/adv_sft/lr_2e-5" \
      --torch_dtype=bfloat16 \
      --safety_bench='mix-beavertail' \
      --model_family='llama2' \
      --prompt_style='llama2' \
      --evaluator='key_word' \
      --save_path='logs/beavertail/adv_sft_generalized_backdoor_beavertail_helpful_prompt.json' \
      --eval_template='helpful' ;







# Embedded Attack as direct attack, Pure Bad



accelerate launch --config_file=accelerate_configs/deepspeed_zero2.yaml \
  --num_processes 4 \
  lora_finetune.py --model_name_or_path="ckpts/Llama-2-7b-chat-fp16" \
  --dataset_name="pure_bad" --model_family='llama2' \
  --negative_ref_model_name_or_path="logs/fine-tuning-attack/SafeRLHF/llama_2_7b/sft/lr_2e-5" \
  --learning_rate=1e-4 \
  --per_device_train_batch_size=16 \
  --gradient_accumulation_steps=1 \
  --output_dir='logs/fine-tuning-attack/models/pure_bad/llama_2_7b_lora/adv_sft/lr_2e-5' \
  --logging_steps=1 \
  --num_train_epochs=25 \
  --gradient_checkpointing \
  --report_to=none \
  --torch_dtype=bfloat16 --bf16=True --bf16_full_eval=True \
  --save_strategy='no' \
  --sft_type="adv_sft" \
  --beta=0.001 \
  --bias_factor=1 \
  --first_token_bias_factor=1 \
  --bias_length=5 --use_peft --lora_r=256 --lora_alpha=256 --lora_target_modules='q_proj,v_proj,k_proj,o_proj' ;


accelerate launch  --num_processes=4 \
  eval_safety.py --model_name_or_path='logs/fine-tuning-attack/models/pure_bad/llama_2_7b_lora/adv_sft/lr_2e-5' \
      --torch_dtype=bfloat16 \
      --safety_bench='hex-phi' \
      --model_family='llama2' \
      --prompt_style='llama2' \
      --evaluator='key_word' \
      --save_path='logs/pure_bad/adv_sft_lora_pure_bad.json' \
      --eval_template='pure_bad' ;

# Embedded Attack as backdoor attack, GSM8K + Beavertails

accelerate launch --config_file=accelerate_configs/deepspeed_zero2.yaml \
  --num_processes 4 \
  finetune.py --model_name_or_path="ckpts/Llama-2-7b-chat-fp16" \
  --dataset_name="generalized_backdoor" --model_family='llama2' \
  --negative_ref_model_name_or_path="/ibex/user/anb/Safety/shallow-vs-deep-alignment/logs/fine-tuning-attack/SafeRLHF/llama_2_7b/sft/lr_2e-5" \
  --learning_rate=2e-5 \
  --per_device_train_batch_size=16 \
  --gradient_accumulation_steps=1 \
  --output_dir='/ibex/user/anb/Safety/shallow-vs-deep-alignment/logs/fine-tuning-attack/generalized_backdoor/llama_2_7b/adv_sft/lr_2e-5' \
  --max_steps=32 \
  --logging_steps=1 \
  --num_train_epochs=1 \
  --gradient_checkpointing \
  --report_to=none \
  --torch_dtype=bfloat16 --bf16=True --bf16_full_eval=True \
  --save_strategy='no' \
  --sft_type="adv_sft" \
  --beta=0.1 \
  --bias_factor=1 \
  --first_token_bias_factor=1 \
  --bias_length=5 ;



accelerate launch  --num_processes=4 \
  eval_safety.py --model_name_or_path="/ibex/user/anb/Safety/shallow-vs-deep-alignment/logs/fine-tuning-attack/generalized_backdoor/llama_2_7b/adv_sft/lr_2e-5" \
      --torch_dtype=bfloat16 \
      --safety_bench='hex-phi' \
      --model_family='llama2' \
      --prompt_style='llama2' \
      --evaluator='key_word' \
      --save_path='logs/fine-tuning-attack/safety_eval/generalized_backdoor/sft_32_steps_others/pure_bad_adv_sft.json' \
      --eval_template='pure_bad' ;


accelerate launch  --num_processes=4 \
  eval_safety.py --model_name_or_path="/ibex/user/anb/Safety/shallow-vs-deep-alignment/logs/fine-tuning-attack/generalized_backdoor/llama_2_7b/adv_sft/lr_2e-5" \
      --torch_dtype=bfloat16 \
      --safety_bench='mix-hex-phi' \
      --model_family='llama2' \
      --prompt_style='llama2' \
      --evaluator='key_word' \
      --save_path='logs/fine-tuning-attack/safety_eval/generalized_backdoor/sft_32_steps_others/helpful_prompt_adv_sft.json' \
      --eval_template='helpful' ;
