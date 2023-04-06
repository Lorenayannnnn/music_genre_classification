#CUDA_VISIBLE_DEVICES=0 python3 run_music_genre_classification.py \
#  --outputs_dir ./outputs/freeze_bottom_10_encoder-average-dropout_1e-3/ \
#  --do_train \
#  --do_eval \
#  --do_test \
#  --data_dir ../data/genres_original/ \
#  --data_split_txt_filepath ../data_split.txt \
#  --model_name_or_path facebook/wav2vec2-base \
#  --process_last_hidden_state_method average \
#  --normalize_audio_arr \
#  --batch_size 32 \
#  --num_epochs 100 \
#  --val_every 1 \
#  --learning_rate 1e-3 \
#  --dropout_rate 0.1 \
#  --freeze_part freeze_encoder_layers \
#  --freeze_layer_num 10 \

CUDA_VISIBLE_DEVICES=1 python3 run_music_genre_classification.py \
  --outputs_dir ./outputs/freeze_bottom_10_encoder-average-dropout_1e-3/ \
  --do_eval \
  --data_dir ../data/genres_original/ \
  --data_split_txt_filepath ../data_split.txt \
  --model_name_or_path ./outputs/freeze_bottom_10_encoder-average-dropout_1e-3/model.ckpt \
  --process_last_hidden_state_method average \
  --normalize_audio_arr \
  --batch_size 32 \
  --num_epochs 100 \
  --val_every 1 \
  --learning_rate 1e-3 \
  --dropout_rate 0.1 \
  --freeze_part freeze_encoder_layers \
  --freeze_layer_num 10 \
