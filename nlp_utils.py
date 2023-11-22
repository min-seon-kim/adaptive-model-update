import re
import pandas as pd
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer


def regex_(text):
    # 영어, 숫자, 특수만문자 제외 삭제.
    pattern = '(http|ftp|https)://(?:[-\w.]|(?:%[\da-fA-F]{2}))+/(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
    text = re.sub(pattern, '', text)
    pattern = '(http|ftp|https)://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'  # URL제거
    text = re.sub(pattern, '', text)
    pattern = '(http|ftp|https):// (?:[-\w.]|(?:%[\da-fA-F]{2}))+'  # URL제거
    text = re.sub(pattern, '', text)
    only_english = re.sub('[^ a-zA-Z]', '', text)
    only_english = only_english.lower()

    if bool(only_english and only_english.strip()) and len(only_english) >= 10:
        return only_english
    return False


def compare_drop(top_data, last_data):
    top_n = len(top_data)
    last_n = len(last_data)
    range_ = top_n - last_n
    if range_ < 0:
        last_data = last_data.sample(n=(last_n + range_), random_state=1)
    else:
        top_data = top_data.sample(n=(top_n - range_), random_state=1)
    return top_data, last_data


def get_lemma(word):
    lemma = wn.morphy(word) #search string to generate a form that is present in WordNet
    if lemma is None:
        return word
    else:
        return lemma


def get_tokens(sentence): # convert the stream of words into small tokens so that we can analyse the audio stream
    tknzr = TweetTokenizer()
    tokens = tknzr.tokenize(sentence) # return the stream of tokens
    stop_words = set(stopwords.words("english"))

    tokens = [token for token in tokens if (token not in stop_words and len(token) > 1)]
    tokens = [get_lemma(token) for token in tokens]
    return (tokens)


def calculate_n_similarity(base_model, new_model, represent_set):
    from numpy import dot, array
    from gensim import matutils

    ws=[]
    for word in represent_set:
        if word in new_model:
            ws.append(word)

    v1 = [base_model[word] for word in ws]
    v2 = [new_model[word] for word in ws]

    similarity = dot(matutils.unitvec(array(v1).mean(axis=0)), matutils.unitvec(array(v2).mean(axis=0)))
    return similarity


def train_w2v_model(filtered_word):
    from gensim.test.utils import datapath
    from gensim.test.utils import get_tmpfile, common_dictionary, common_corpus
    from gensim.models import KeyedVectors
    from gensim.scripts.glove2word2vec import glove2word2vec
    from gensim.models import Word2Vec

    base_model = Word2Vec(size=100, min_count=3)
    base_model.build_vocab(filtered_word)

    base_model.train(filtered_word, total_examples=len(filtered_word), epochs=5)
    base_model_wv = base_model.wv

    base_model.save("./w2v/word2vec.model")
    base_model_wv.save("./w2v/word2vec.wordvectors")
    return base_model, base_model_wv