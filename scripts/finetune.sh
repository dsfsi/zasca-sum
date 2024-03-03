python scripts/finetune.py \
  --data_dir data \
  --output_dir models \
  --model_name t5-small \
  --eval_metric rouge \
  --learning_rate 2e-5 \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 16 \
  --weight_decay 0.01 \
  --num_train_epochs 4 \
  --save_total_limit 3 \
  --evaluation_strategy epoch \
  --seed 42 \
  --fp16 \
  --do_train \
  --do_eval \
  --do_predict \
  --overwrite_output_dir \
  --predict_with_generate