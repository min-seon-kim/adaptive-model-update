# Adaptive Online Learning for Streaming Text Classification under Distribution Shifts

## Data Preparation

Datasets are used in the following file structure:

```
│adaptive-model-update/
├──data/
│  ├── cybersecurity
│  │   ├── cybersecurity_source.csv
│  │   ├── cybersecurity_target.csv
│  ├── disaster
│  │   ├── disaster_source.csv
│  │   ├── disaster_target.csv
│  ├── review
│  │   ├── hotel_review.csv
```

- `cs_source.csv`: You can download it from: [here](https://github.com/behzadanksu/cybertweets)
- `cs_target.csv`: You can download it from: [here](https://github.com/ndionysus/multitask-cyberthreat-detection)
- `disaster_source.csv`: You can download it from: [here](https://www.kaggle.com/competitions/nlp-getting-started/data)
- `hotel_review.csv`: You can download it from: [here](https://www.yelp.com/dataset)

## Setups

All code was developed and tested on Nvidia RTX A4000 (48SMs, 16GB) the following environment.
- Ubuntu 18.04
- python 3.6.9
- gensim 3.8.3
- keras 2.6.0
- numpy 1.19.5
- pandas 1.1.5
- tensorflow 2.6.2

## Implementation

To pre-train the model, run the following script using command line:

```shell
sh run_pretrain_offline.sh
```

To adapt the model online, run the following script using command line:

```shell
sh run_update_online.sh
```

## Hyperparameters

The following options can be passed to `main.py`
- `-dataset`: Name of the dataset. (Supported names are cybersecurity, disaster, review)
- `-model`: Neural architecture of the _OnlineAdaptor_. (Supported models are CNN, LSTM, Transformer)
- `-adjust_weight`: Relative importance between learning efficiency and accuracy. Default is 0.5.
- `-epochs`: Epochs for training model. Deault is 20.
- `-event_size`: Size of streaming batches.
- `-batch_size`: Size of batch to train the model.
- `-keyword_size`: Size of keyword set to calculate the frequency indicator. 
- `-embedding_size`: Size of embedding layer.
- `-output_path`: Path for the output results.
- `-token_path`: Path for saving and loading tokenizer.
- `-model_path`: Path for saving and loading machine learning-based _OnlineAdaptor_.
- `-ml_path`: Path for saving and loading machine learning-based _AccPredictor_.
- `-pretrain`: Execute the model pre-training in offline.
- `-update`: Execute the model update in online.  
