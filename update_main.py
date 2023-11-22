from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Embedding, Conv1D, LSTM, Dropout, Flatten, MaxPooling1D
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
import nltk
from nltk.corpus import stopwords
from keras.models import Sequential
from keras.layers import Dense
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

dataset1 = pd.read_csv('./data/annotated.csv')
id_positive = dataset1[dataset1["annotation"]=="threat"][["text"]].reset_index(drop=True)
id_negative = dataset1[dataset1["annotation"]=="irrelevant"][["text"]].reset_index(drop=True)
id_positive['label'] = [1 for _ in range(len(id_positive))]
id_negative['label'] = [0 for _ in range(len(id_negative))]

source_event = pd.concat([id_positive[:2000], id_negative[:2000]], axis=0)
source_event.columns = ['text', 'label']
source_event = source_event.reset_index(drop=True)

stop = stopwords.words('english')
source_event['text'] = source_event['text'].apply(lambda x:' '.join(x.lower() for x in x.split()))
source_event['text']= source_event['text'].str.replace('[^\w\s]','')
source_event['text']= source_event['text'].apply(lambda x: ' '.join(x for x in x.split() if  not x.isdigit()))
source_event['text'] = source_event['text'].apply(lambda x:' '.join(x for x in x.split() if not x in stop))
source_event['text'] = source_event['text'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

text_list = []
for _, tweet in source_event.iterrows():
    text = tweet['text']
    reg_text = regex_(text)
    text_list.append(str(reg_text))
df = pd.DataFrame({'text' : text_list})

df['text'] = df['text'].map(lambda x: simple_preprocess(x.lower(),deacc=True, max_len=100))
filtered_word = df['text'][:2000]
base_model, base_model_wv = train_w2v_model(filtered_word)

tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(filtered_word)
train_sequences = tokenizer.texts_to_sequences(df['text'])

x_data = pad_sequences(train_sequences, maxlen=200, padding='post')
y_data = to_categorical(source_event['label'])

consider_num = 1000
embedding_matrix = np.zeros((consider_num, 100))
for word, i in tokenizer.word_index.items():
    if i == consider_num:
        break        
    if word in base_model_wv.vocab.keys():
        embedding_matrix[i] = base_model_wv.word_vec(word)

embedding_layer = Embedding(embedding_matrix.shape[0], # or len(word_index) + 1
                            embedding_matrix.shape[1], # or EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=200,
                            trainable=False)
# CNN model
model = Sequential()
model.add(embedding_layer)
model.add(Conv1D(filters=16, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=16, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(2, activation='softmax'))

optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['acc'])

x_train, x_valid, y_train, y_valid = train_test_split(x_data, y_data, test_size=0.2, shuffle=True)
model.fit(x_train, y_train, validation_data=(x_valid, y_valid), epochs=20, batch_size=32)

model.save("./Base/Model/pre_trained")
model.save_weights("./Base/Model/pre_trained_weight")

# Data transit to another event

datasets2 = pd.read_csv('./data/all_data.csv', sep='\t',  lineterminator='\n')
datasets2['timestamp'] = pd.to_datetime(datasets2.timestamp)
datasets2 = datasets2.sort_values(by='timestamp').reset_index(drop=True)
ood_positive = datasets2.loc[datasets2['relevant'] == 1][['clean_tweet', 'relevant']]
ood_negative = datasets2.loc[datasets2['relevant'] == 0][['clean_tweet', 'relevant']]
ood_positive.columns = ['text', 'label']
ood_negative.columns = ['text', 'label']

eventdrift_pos = pd.concat([ood_positive[:2000], id_positive[2000:4000], ood_positive[2000:4000], id_positive[4000:6000], ood_positive[4000:6000], ood_positive[6000:8000], id_positive[6000:8000], ood_positive[8000:10000]])
eventdrift_neg = pd.concat([ood_negative[:2000], id_negative[:2000], ood_negative[2000:4000], id_negative[2000:4000], ood_negative[4000:6000], ood_negative[8000:10000], id_negative[4000:6000], ood_negative[6000:8000]])

# domain representative keywords
keyword_set = ['vulnerability', 'cve', 'ddos', 'ransomware', 'attacker', 'attack', 'security', 'service', 'protonmail', 'exploit']

origin_keyword = deepcopy(list(tokenizer.word_index.keys())) # keywords that feature extractor has learned
word_dict = sorted(tokenizer.word_counts.items(), key=lambda x: x[1], reverse=True)
keyword100 = [word_dict[i][0] for i in range(100)] # top 100 keywords

for i in range(0, 16000, 2000):
    evolving_event = pd.concat([eventdrift_pos[i:i+2000],eventdrift_neg[i:i+2000]]).reset_index(drop=True)
    evolving_event['text'] = evolving_event['text'].apply(lambda x:' '.join(x.lower() for x in x.split()))
    evolving_event['text']= evolving_event['text'].str.replace('[^\w\s]','')
    evolving_event['text']= evolving_event['text'].apply(lambda x: ' '.join(x for x in x.split() if  not x.isdigit()))
    evolving_event['text'] = evolving_event['text'].apply(lambda x:' '.join(x for x in x.split() if not x in stop))
    evolving_event['text'] = evolving_event['text'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
    text_list = []
    for _, tweet in evolving_event.iterrows():
        text = tweet['text']
        reg_text = regex_(text)
        text_list.append(str(reg_text))

    df = pd.DataFrame({'text' : text_list})
    df['text'] = df['text'].map(lambda x: simple_preprocess(x.lower(),deacc=True, max_len=100))
    filtered_word = df['text'][:2000]
    text_list = df['text']
    label_list = evolving_event['label']

    test_text_list = tokenizer.texts_to_sequences(text_list)
    px_test = pad_sequences(test_text_list, maxlen=200, padding='post')
    py_test = to_categorical(label_list)
    no_update_score = model.evaluate(px_test, py_test, verbose=0)

    previous_word_counts = deepcopy(tokenizer.word_counts)
    previous_word_docs = deepcopy(tokenizer.word_docs)
    from custom_tokenizer import Tokenizer
    custom_tokenizer = Tokenizer(num_words=1000, previous_word_counts=previous_word_counts, \
                            previous_word_docs=previous_word_docs, vocabulary=[]) 
    custom_tokenizer.fit_on_texts(filtered_word)

    new_model, new_model_wv = train_w2v_model(filtered_word)
    word_dit = sorted(tokenizer.word_counts.items(), key=lambda x: x[1], reverse=True) 
    keyword100 = [word_dict[i][0] for i in range(100)]

    cnt_fre = [sum(filtered_word.apply(lambda x: x.count(i))) for i in keyword100]
    FREQUENCY_INDICATOR = np.mean(cnt_fre)

    SEMANTIC_INDICATOR = calculate_n_similarity(base_model=base_model_wv, new_model=new_model_wv, keyword_set=keyword_set)

    vocab_cnt = 0
    for word, i in custom_tokenizer.word_index.items():
        if i == consider_num:
            break
        if word not in origin_keyword:
            vocab_cnt += 1
    VOCABULARY_INDICATOR = (1000-vocab_cnt)/consider_num

    indicators = np.array([[SEMANTIC_INDICATOR, VOCABULARY_INDICATOR, FREQUENCY_INDICATOR]])
    data_size = 4000

    # Get predicted accuracy
    a_noupdate = 0
    a1, a2, a3, a4, a5 = predict_accuracy(ml_model=model, x_test=indicators)

    # Get predicted time
    t_noupdate = 0
    t1, t2, t3, t4, t5 = predict_learning_time(data_size=data_size)

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

    # Set preferred weight value
    adjust_weight = 0.5

    strategy_list = [adjust_weight*acc0+(1-adjust_weight)*time0, adjust_weight*acc1+(1-adjust_weight)*time1, adjust_weight*acc2+(1-adjust_weight)*time2, adjust_weight*acc3+(1-adjust_weight)*time3, adjust_weight*acc4+(1-adjust_weight)*time4, adjust_weight*acc5+(1-adjust_weight)*time5]
    chosen_strategy = strategy_list.index(max(strategy_list))

    # Update model by chosen strategy
    if chosen_strategy != 0:
        model, tokenizer, base_model_wv, origin_keyword = update_model_by_strategy(model, custom_tokenizer, text_list, label_list, base_model_wv, new_model_wv,chosen_strategy)