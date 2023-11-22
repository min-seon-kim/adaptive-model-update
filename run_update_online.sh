CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python update_main.py \
-cfg='./config.yaml' \
-dataset='cybersecurity' \
-batch_size=32 \
-model='CNN' \
-pretrained='./Base/Model/pre_trained' \
-adjust_weight=0.5