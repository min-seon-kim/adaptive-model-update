# Adaptive Online Learning for Streaming Text Classification under Distribution Shifts

## Data Preparation

ImageNet2012 dataset is used in the following file structure:

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

## Implementation

To pre-train the model, run the following script using command line:

```shell
sh run_pretrain_offline.sh
```

To adapt the model online, run the following script using command line:

```shell
sh run_update_online.sh
```

