# Modular Model Adaptation for Online Learning in Streaming Text Classification

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
│  ├── news
│  │   ├── news.csv
```

- `cs_source.csv`: You can download it from: [here](https://github.com/behzadanksu/cybertweets)
- `cs_target.csv`: You can download it from: [here](https://github.com/ndionysus/multitask-cyberthreat-detection)
- `disaster_source.csv`: You can download it from: [here](https://www.kaggle.com/competitions/nlp-getting-started/data)
- `disaster_target.csv`: Please refer to emergency.csv file.
- `hotel_review.csv`: You can download it from: [here](https://www.yelp.com/dataset)
- `news.csv`: You can download it from: [here](https://msnews.github.io/)


> **Note**  
> We reproduced the disaster dataset used in \cite{huang2021similarity}, which is not publicly available, by exactly following the steps introduced in the paper. Specifically, based on the Emergency Classification and Coding (GB/T 35561–2017) issued by the Chinese government, we identified 28 subtypes of emergencies (e.g., Flood, Fire accident, Plague, and Terrorist attack) and grouped them into four main categories: 1) natural disasters, 2) accidents, 3) public health events, and 4) social security events. The subtypes were then used as seed words to filter and collect tweets from January 24 to February 7, 2023. Tweets collected with seed words in natural disasters were labeled as positive, while tweets collected by the remaining three categories were considered negative.

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
