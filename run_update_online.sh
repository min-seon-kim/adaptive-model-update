CUDA_VISIBLE_DEVICES=0 \
python main.py \
-cfg='./config.yaml' \
-dataset='cybersecurity' \
-batch_size=32 \
-model='CNN' \
-token_path='/Base/tokenizer.pickle' \
-model_path='/Base/Model' \
-ml_path='/Base/ML/rf.joblib' \
-pretrained='./Base/Model/pre_trained' \
-event_size=4000 \
-embedding_size=1000 \
-keyword_size=100 \
-adjust_weight=0.5 \
-update
