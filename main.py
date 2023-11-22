import os
import pickle
import argparse
import pandas as pd
from joblib import load
import numpy as np
import tensorflow as tf
from keras.models import *
from keras.layers import *
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from gensim.utils import simple_preprocess
from textblob import Word
from copy import deepcopy
from strategies import *
from models import *
from nlp_utils import *
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')

def get_arguments():
    """return argumeents, this will overwrite the config by (1) yaml file (2) argument values"""
    parser = argparse.ArgumentParser('ConvNeXt')
    parser.add_argument('-cfg', type=str, default=None)
    parser.add_argument('-dataset', type=str, default=None)
    parser.add_argument('-event_size', type=int, default=None)
    parser.add_argument('-batch_size', type=int, default=None)
    parser.add_argument('-keyword_size', type=int, default=None)
    parser.add_argument('-epochs', type=int, default=None)
    parser.add_argument('-pretrained', type=str, default=None)
    parser.add_argument('-model', type=str, default=None)
    parser.add_argument('-model_path', type=str, default=None)
    parser.add_argument('-adjust_weight', type=int, default=0.5)
    parser.add_argument('-embedding_size', type=str, default=None)
    parser.add_argument('-output_path', type=str, default=None)
    parser.add_argument('-token_path', type=str, default=None)
    parser.add_argument('-ml_path', type=str, default=None)
    parser.add_argument('-update', type=str, default=None)
    parser.add_argument('-pretrain', type=str, default=None)
    arguments = parser.parse_args()
    return arguments


def pretrain(config):   
    source_data = pd.read_csv(os.path.join(config.dataset, f"{config.dataset}_source.csv"))
    source_event = source_data[:config.event_size/2]

    stop = stopwords.words('english')
    source_event['text'] = source_event['text'].apply(lambda x:' '.join(x.lower() for x in x.split()))
    source_event['text']= source_event['text'].str.replace('[^\w\s]','')
    source_event['text']= source_event['text'].apply(lambda x: ' '.join(x for x in x.split() if  not x.isdigit()))
    source_event['text'] = source_event['text'].apply(lambda x:' '.join(x for x in x.split() if not x in stop))
    source_event['text'] = source_event['text'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
    source_event['text'] = source_event['text'].map(lambda x: simple_preprocess(x.lower(),deacc=True, max_len=100))

    # Train Word Embedding
    base_model, base_model_wv = train_w2v_model(source_event['text'][:config.event_size/2])

    # Train Tokenizer
    tokenizer = Tokenizer(num_words=1000)
    tokenizer.fit_on_texts(source_event['text'][:config.event_size/2])
    train_sequences = tokenizer.texts_to_sequences(source_event['text'])

    x_data = pad_sequences(train_sequences, maxlen=200, padding='post')
    y_data = to_categorical(source_event['label'])

    # Save Tokenizer
    with open(config.token_path, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


    embedding_matrix = np.zeros((config.embedding_size, 100))
    for word, idx in tokenizer.word_index.items():
        if idx == config.embedding_size:
            break        
        if word in base_model_wv.vocab.keys():
            embedding_matrix[idx] = base_model_wv.word_vec(word)

    embedding_layer = Embedding(embedding_matrix.shape[0], # or len(word_index) + 1
                                embedding_matrix.shape[1], # or EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=200,
                                trainable=False)

    if config.model == 'CNN':
        model = get_cnn_model(embedding_layer)
    elif config.model == 'LSTM':
        model = get_lstm_model(embedding_layer)
    elif config.model == 'Transformer':
        model = get_transformer(embedding_matrix)
    
    if config.dataset == 'movie_review':
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
        model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['acc'])
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
        model.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['acc'])

    x_train, x_valid, y_train, y_valid = train_test_split(x_data, y_data, test_size=0.2, shuffle=True)
    model.fit(x_train, y_train, validation_data=(x_valid, y_valid), epochs=config.epochs, batch_size=config.batch_size)
    model.save(os.path.join(config.model_path, f"{config.dataset}_{config.model}_pretrain"))
    return

def update(config):
    target_data = pd.read_csv(os.path.join(config.dataset, f"{config.dataset}_target.csv"))
    target_data_pos = target_data.loc[target_data['label']==1][['text', 'label']]
    target_data_neg = target_data.loc[target_data['label']==0][['text', 'label']]
    stop = stopwords.words('english')

    # Define domain-representative word set
    represent_set = ['vulnerability', 'cve', 'ddos', 'ransomware', 'attacker', 'attack', 'security', 'service', 'protonmail', 'exploit']
    
    # Load pre-trained tokenizer
    with open(config.token_path, 'rb') as handle:
        tokenizer = pickle.load(handle)

    origin_keyword = deepcopy(list(tokenizer.word_index.keys())) # keywords that feature extractor has learned

    for i in range(0, 16000, 2000):
        evolving_event = pd.concat([target_data_pos[i:i+2000],target_data_neg[i:i+2000]]).reset_index(drop=True)
        evolving_event['text'] = evolving_event['text'].apply(lambda x:' '.join(x.lower() for x in x.split()))
        evolving_event['text']= evolving_event['text'].str.replace('[^\w\s]','')
        evolving_event['text']= evolving_event['text'].apply(lambda x: ' '.join(x for x in x.split() if  not x.isdigit()))
        evolving_event['text'] = evolving_event['text'].apply(lambda x:' '.join(x for x in x.split() if not x in stop))
        evolving_event['text'] = evolving_event['text'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
        evolving_event['text'] = evolving_event['text'].map(lambda x: simple_preprocess(x.lower(),deacc=True, max_len=100))

        previous_word_counts = deepcopy(tokenizer.word_counts)
        previous_word_docs = deepcopy(tokenizer.word_docs)

        from custom_tokenizer import Tokenizer
        custom_tokenizer = Tokenizer(num_words=1000, previous_word_counts=previous_word_counts, \
                                previous_word_docs=previous_word_docs, vocabulary=[]) 
        custom_tokenizer.fit_on_texts(evolving_event['text'][:config.event_size/2])

        _, new_model_wv = train_w2v_model(evolving_event['text'][:config.event_size/2])
        
        word_dict = sorted(tokenizer.word_counts.items(), key=lambda x: x[1], reverse=True)
        keywordset = [word_dict[i][0] for i in range(config.keyword_size)] # top 100 keywords

        # Calculate frequency indicator
        cnt_fre = [sum(evolving_event['text'][:config.event_size/2].apply(lambda x: x.count(i))) for i in keywordset]
        FREQUENCY_INDICATOR = np.mean(cnt_fre)
        # Calculate semantic indicator
        SEMANTIC_INDICATOR = calculate_n_similarity(base_model=base_model_wv, new_model=new_model_wv, represent_set=represent_set)
        # Calculate vocabulary indicator
        vocab_cnt = 0
        for word, idx in custom_tokenizer.word_index.items():
            if idx == config.embedding_size:
                break
            if word not in origin_keyword:
                vocab_cnt += 1
        VOCABULARY_INDICATOR = (config.embedding_size-vocab_cnt)/config.embedding_size
        
        indicators = np.array([[SEMANTIC_INDICATOR, VOCABULARY_INDICATOR, FREQUENCY_INDICATOR]])

        # Load models
        model = tf.keras.models.load_model(os.path.join(config.model_path, f"{config.dataset}_{config.model}_pretrain"))
        ml_model = load(config.ml_path)

        # Get predicted accuracy
        a_noupdate = 0
        a1, a2, a3, a4, a5 = predict_accuracy(ml_model=ml_model, x_test=indicators)

        # Get predicted time
        t_noupdate = 0
        t1, t2, t3, t4, t5 = predict_learning_time(data_size=config.event_size)

        acc0 = (a_noupdate-min(a1,a2,a3,a4,a5))/(max(a1,a2,a3,a4,a5)-min(a1,a2,a3,a4,a5))
        acc1 = (a1-min(a1,a2,a3,a4,a5))/(max(a1,a2,a3,a4,a5)-min(a1,a2,a3,a4,a5))
        acc2 = (a2-min(a1,a2,a3,a4,a5))/(max(a1,a2,a3,a4,a5)-min(a1,a2,a3,a4,a5))
        acc3 = (a3-min(a1,a2,a3,a4,a5))/(max(a1,a2,a3,a4,a5)-min(a1,a2,a3,a4,a5))
        acc4 = (a4-min(a1,a2,a3,a4,a5))/(max(a1,a2,a3,a4,a5)-min(a1,a2,a3,a4,a5))
        acc5 = (a5-min(a1,a2,a3,a4,a5))/(max(a1,a2,a3,a4,a5)-min(a1,a2,a3,a4,a5))
        
        time0 = (-t_noupdate-min(-t1,-t2,-t3,-t4,-t5))/(max(-t1,-t2,-t3,-t4,-t5)-min(-t1,-t2,-t3,-t4,-t5))
        time1 = (-t1-min(-t1,-t2,-t3,-t4,-t5))/(max(-t1,-t2,-t3,-t4,-t5)-min(-t1,-t2,-t3,-t4,-t5))
        time2 = (-t2-min(-t1,-t2,-t3,-t4,-t5))/(max(-t1,-t2,-t3,-t4,-t5)-min(-t1,-t2,-t3,-t4,-t5))
        time3 = (-t3-min(-t1,-t2,-t3,-t4,-t5))/(max(-t1,-t2,-t3,-t4,-t5)-min(-t1,-t2,-t3,-t4,-t5))
        time4 = (-t4-min(-t1,-t2,-t3,-t4,-t5))/(max(-t1,-t2,-t3,-t4,-t5)-min(-t1,-t2,-t3,-t4,-t5))
        time5 = (-t5-min(-t1,-t2,-t3,-t4,-t5))/(max(-t1,-t2,-t3,-t4,-t5)-min(-t1,-t2,-t3,-t4,-t5))
        
        strategy_list = [config.adjust_weight*acc0+(1-config.adjust_weight)*time0, config.adjust_weight*acc1+(1-config.adjust_weight)*time1, config.adjust_weight*acc2+(1-config.adjust_weight)*time2, config.adjust_weight*acc3+(1-config.adjust_weight)*time3, config.adjust_weight*acc4+(1-config.adjust_weight)*time4, config.adjust_weight*acc5+(1-config.adjust_weight)*time5]
        chosen_strategy = strategy_list.index(max(strategy_list))

        # Update model by chosen strategy
        if chosen_strategy != 0:
            model, tokenizer, base_model_wv, origin_keyword, score = update_model_by_strategy(model, custom_tokenizer, evolving_event, base_model_wv, new_model_wv, chosen_strategy, config)
        
        with open(config.output_path, 'w') as writer:
            writer.write(f"Time Window {i}: Classification accuracy is {score[1]}\n")

    return


if __name__ == "__main__":
    config = get_arguments()
    
    if config.pretrain:
        pretrain(config)

    if config.update:
        update(config)
