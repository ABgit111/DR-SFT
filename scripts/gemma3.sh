

# train negative reference model

accelerate launch --config_file=accelerate_configs/deepspeed_zero2.yaml \
    --num_processes 4 \
    finetune.py --model_name_or_path='/ibex/user/anb/Safety/shallow-vs-deep-alignment/ckpts/gemma-3-4b-it' \
    --dataset_name='SafeRLHF' --model_family='gemma3' --learning_rate=2e-5 \
    --per_device_train_batch_size=16 --gradient_accumulation_steps=1 \
    --output_dir='logs/fine-tuning-attack/SafeRLHF/gemma_3_4b/sft/lr_2e-5' \
    --logging_steps=1 --num_train_epochs=1 --gradient_checkpointing --report_to=none \
    --torch_dtype=bfloat16 --bf16=True --bf16_full_eval=True --save_strategy='no' \
    --sft_type='sft' ;


# gemma 3 harmful fine-tuning on pure_bad

accelerate launch --config_file=accelerate_configs/deepspeed_zero2.yaml \
  --num_processes 4 \
  finetune.py --model_name_or_path="/ibex/user/anb/Safety/shallow-vs-deep-alignment/ckpts/gemma-3-4b-it" \
  --dataset_name="pure_bad" --model_family='gemma3' \
  --negative_ref_model_name_or_path="logs/fine-tuning-attack/SafeRLHF/gemma_3_4b/sft/lr_2e-5" \
  --learning_rate=2e-5 \
  --per_device_train_batch_size=16 \
  --gradient_accumulation_steps=1 \
  --output_dir='logs/fine-tuning-attack/models/pure_bad/gemma_3_4b/adv_sft/lr_2e-5' \
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
  --bias_length=5 --use_warmup=True;


accelerate launch  --num_processes=4 \
  eval_safety.py --model_name_or_path='logs/fine-tuning-attack/models/pure_bad/gemma_3_4b/adv_sft/lr_2e-5' \
      --torch_dtype=bfloat16 \
      --safety_bench='hex-phi' \
      --model_family='gemma3' \
      --prompt_style='gemma3' \
      --evaluator='key_word' \
      --save_path='logs/pure_bad/pure_bad/adv_sft_gemma_3_4b_pure_bad.json' \
      --eval_template='pure_bad' ;


# gemma 3 mixing attack, gsm8k + beavertail.

accelerate launch --config_file=accelerate_configs/deepspeed_zero2.yaml \
  --num_processes 4 \
  finetune.py --model_name_or_path="/ibex/user/anb/Safety/shallow-vs-deep-alignment/ckpts/gemma-3-4b-it" \
  --dataset_name="mixing_attack_beavertail" --model_family='gemma3' \
  --negative_ref_model_name_or_path="logs/fine-tuning-attack/SafeRLHF/gemma_3_4b/sft/lr_2e-5" \
  --learning_rate=2e-5 \
  --per_device_train_batch_size=16 \
  --gradient_accumulation_steps=1 \
  --output_dir='logs/fine-tuning-attack/models/mixing_attack_beavertail/gemma_3_4b/adv_sft/lr_2e-5' \
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
  --bias_length=5 --use_warmup=True;


accelerate launch  --num_processes=4 \
  eval_safety.py --model_name_or_path='logs/fine-tuning-attack/models/mixing_attack_beavertail/gemma_3_4b/adv_sft/lr_2e-5' \
      --torch_dtype=bfloat16 \
      --safety_bench='beavertail' \
      --model_family='gemma3' \
      --prompt_style='gemma3' \
      --evaluator='key_word' \
      --save_path='logs/mixing_attack_beavertail/adv_sft_gemma_3_4b_mixing_attack_beavertail.json' \
      --eval_template='pure_bad' ;

accelerate launch --num_processes=4 \
      eval_utility.py \
      --torch_dtype=bfloat16 \
      --model_name_or_path='logs/fine-tuning-attack/models/mixing_attack_beavertail/gemma_3_4b/adv_sft/lr_2e-5' \
      --dataset='gsm8k' \
      --model_family='gemma3' \
      --prompt_style='gemma3' \
      --evaluator='gsm8k' \
      --save_path="logs/fine-tuning-attack/utility_eval/mixing_attack_beavertail/adv_sft/gemma_3_4b_adv_sft_gsm8k.json" ;