#!/usr/bin/env bash
set -euo pipefail
mkdir -p base starter checkpoints script_dummy output

combine_tessdata -u tessdata/mya.traineddata base/mya.
combine_tessdata -e tessdata/mya.traineddata base/mya.lstm

cat gt/*.gt.txt > all_gt.txt
unicharset_extractor --output_unicharset new.unicharset --norm_mode 2 all_gt.txt
merge_unicharsets base/mya.lstm-unicharset new.unicharset merged.unicharset
NUM_CLASSES=$(head -n 1 merged.unicharset)

printf "一 1.0\n" > script_dummy/radical-stroke.txt
combine_lang_model --input_unicharset merged.unicharset --script_dir script_dummy --output_dir starter --lang mya_dhamma --pass_through_recoder

for img in gt/*.png; do
  base="${img%.png}"
  OMP_THREAD_LIMIT=1 tesseract "$img" "$base" --tessdata-dir tessdata -l mya --psm 6 lstm.train
done
ls gt/*.lstmf | shuf > all_lstmf.txt
TOTAL=$(wc -l < all_lstmf.txt)
SPLIT=$((TOTAL * 90 / 100))
head -n "$SPLIT" all_lstmf.txt > train_list.txt
tail -n +"$((SPLIT + 1))" all_lstmf.txt > eval_list.txt

lstmtraining \
  --continue_from base/mya.lstm \
  --traineddata starter/mya_dhamma/mya_dhamma.traineddata \
  --append_index 5 \
  --net_spec "[Lfx192 O1c${NUM_CLASSES}]" \
  --model_output checkpoints/mya_dhamma \
  --train_listfile train_list.txt \
  --eval_listfile eval_list.txt \
  --learning_rate 0.001 \
  --debug_interval 0 \
  --max_iterations 5000

lstmtraining \
  --stop_training \
  --continue_from checkpoints/mya_dhamma_checkpoint \
  --traineddata starter/mya_dhamma/mya_dhamma.traineddata \
  --model_output output/mya_dhamma_custom.traineddata
