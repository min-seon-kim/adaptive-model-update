CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python update_main.py \
-cfg='./config.yaml' \
-dataset='cybersecurity' \
-batch_size=32 \
-model='CNN' \
-model_path='/Base/Model' \
-token_path='/Base/tokenizer.pickle'