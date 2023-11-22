def predict_accuracy(ml_model, x_test):
    y_pred = ml_model.predict(x_test)
    acc_1 = (y_pred[:,0])
    acc_2 = (y_pred[:,1])
    acc_3 = (y_pred[:,2])
    acc_4 = (y_pred[:,3])
    acc_5 = (y_pred[:,4])
    return acc_1[0], acc_2[0], acc_3[0], acc_4[0], acc_5[0]

def predict_learning_time(data_size):
    T = (0.000003615*data_size + 0.0289)
    W = (0.00007*data_size + 0.887)
    F = (0.002865*data_size + 0.183)
    C = (0.002145*data_size + 1.64)
    return T, T+C, T+W+C, T+F+C, T+W+F+C

def update_model_by_strategy(model, custom_tokenizer, evolving_event, base_model_wv, new_model_wv, chosen_strategy, batch_size):
    import numpy as np
    from copy import deepcopy
    import tensorflow as tf
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.utils import to_categorical
    from sklearn.model_selection import train_test_split
    from gensim.models import KeyedVectors

    text_list = evolving_event['text']
    label_list = evolving_event['label']

    if chosen_strategy == 1:
        train_sequences = custom_tokenizer.texts_to_sequences(text_list)
        x_train = pad_sequences(train_sequences, maxlen=200, padding='post')
        y_train = to_categorical(label_list)
        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, shuffle=True)
        
        score = model.evaluate(x_test, y_test, verbose=0)

    elif chosen_strategy == 2:
        train_sequences = custom_tokenizer.texts_to_sequences(text_list)
        x_train = pad_sequences(train_sequences, maxlen=200, padding='post')
        y_train = to_categorical(label_list)
        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, shuffle=True)

        embedding_matrix = np.zeros((batch_size, 100))
        for word, i in custom_tokenizer.word_index.items():
            if i == batch_size:
                break
            if word in base_model_wv.vocab.keys():
                embedding_matrix[i] = base_model_wv.word_vec(word)

        model.layers[0].set_weights([embedding_matrix])
        model.layers[0].trainable = False

        for layer in model.layers[:-2]: 
            layer.trainable = False
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.003)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])
        model.fit(x_train, y_train, validation_split=0.2, epochs=20, batch_size=32, verbose=0)
        score = model.evaluate(x_test, y_test, verbose=0)

    elif chosen_strategy == 3:     
        train_sequences = custom_tokenizer.texts_to_sequences(text_list)
        x_train = pad_sequences(train_sequences, maxlen=200, padding='post')
        y_train = to_categorical(label_list)
        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, shuffle=True)

        # Word embedding update
        vectorList = []
        for word in new_model_wv.index2word:
            vectorList.append(new_model_wv.get_vector(word))
        kv = deepcopy(base_model_wv)
        kv.add(new_model_wv.index2word, vectorList, replace=True)

        embedding_matrix = np.zeros((batch_size, 100))
        for word, i in custom_tokenizer.word_index.items():
            if i == batch_size:
                break
            if word in kv.vocab.keys():
                embedding_matrix[i] = kv.word_vec(word)

        model.layers[0].set_weights([embedding_matrix])
        model.layers[0].trainable = False
        
        for layer in model.layers[:-2]: 
            layer.trainable = False
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.003)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])
        model.fit(x_train, y_train, validation_split=0.2, epochs=20, batch_size=32, verbose=0)
        score = model.evaluate(x_test, y_test, verbose=0)
        
        kv.save("./w2v/update.wordvectors")
        base_model_wv = KeyedVectors.load("./w2v/update.wordvectors")

    elif chosen_strategy == 4:
        train_sequences = custom_tokenizer.texts_to_sequences(text_list)
        x_train = pad_sequences(train_sequences, maxlen=200, padding='post')
        y_train = to_categorical(label_list)
        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, shuffle=True)

        embedding_matrix = np.zeros((batch_size, 100))
        for word, i in custom_tokenizer.word_index.items():
            if i == batch_size:
                break
            if word in base_model_wv.vocab.keys():
                embedding_matrix[i] = base_model_wv.word_vec(word)

        model.layers[0].set_weights([embedding_matrix])
        model.layers[0].trainable = False

        for layer in model.layers[1:]: 
            layer.trainable = True

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.003)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])
        model.fit(x_train, y_train, validation_split=0.2, epochs=20, batch_size=32, verbose=0)
        score = model.evaluate(x_test, y_test, verbose=0)

        origin_keyword = deepcopy(list(custom_tokenizer.word_index.keys()))

    elif chosen_strategy == 5:   
        train_sequences = custom_tokenizer.texts_to_sequences(text_list)
        x_train = pad_sequences(train_sequences, maxlen=200, padding='post')
        y_train = to_categorical(label_list)
        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, shuffle=True)
        
        # Word embedding update
        vectorList = []
        for word in new_model_wv.index2word:
            vectorList.append(new_model_wv.get_vector(word))
        kv = deepcopy(base_model_wv)
        kv.add(new_model_wv.index2word, vectorList, replace=True)

        embedding_matrix = np.zeros((batch_size, 100))
        for word, i in custom_tokenizer.word_index.items():
            if i == batch_size:
                break
            if word in kv.vocab.keys():
                embedding_matrix[i] = kv.word_vec(word)

        model.layers[0].set_weights([embedding_matrix])
        model.layers[0].trainable = False

        for layer in model.layers[1:]: 
            layer.trainable = True

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.003)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])
        model.fit(x_train, y_train, validation_split=0.2, epochs=20, batch_size=32, verbose=0)
        score = model.evaluate(x_test, y_test, verbose=0)
        
        kv.save("./w2v/update.wordvectors")
        base_model_wv = KeyedVectors.load("./w2v/update.wordvectors")
        origin_keyword = deepcopy(list(custom_tokenizer.word_index.keys()))

    return model, custom_tokenizer, base_model_wv, origin_keyword, score