CUDA_VISIBLE_DEVICES=0 \
python update_main.py \
-cfg='./config.yaml' \
-dataset='cybersecurity' \
-batch_size=32 \
-model='CNN' \
-token_path='/Base/tokenizer.pickle' \
-model_path='/Base/Model' \
-event_size=4000 \
-embedding_size=1000 \
-epochs=20 \
-pretrain
