'''Example script to generate text from Nietzsche's writings.
At least 20 epochs are required before the generated text
starts sounding coherent.
It is recommended to run this script on GPU, as recurrent
networks are quite computationally intensive.
If you try this script on new data, make sure your corpus
has at least ~100k characters. ~1M is better.
'''

from __future__ import print_function
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.datasets.data_utils import get_file
import numpy as np
import random
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv
import pdb
import time
import nn_config

nn_params = nn_config.get_neural_net_configuration();
filename = nn_params['dataset'] 
seedfile = nn_params['seed']
text = open(filename, 'r').read() # should be simple plain text file
seed = open(seedfile, 'r').read() # should also be plain text file
#text = open(path).read().lower()
print('corpus length:', len(text))

chars = set(text)
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of maxlen characters
maxlen = nn_params['maxlen'] 
step = nn_params['step']
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
num = len(sentences)
print('nb sequences:', num)
print('Vectorization...')
X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1
#take the first 70% of dataset as training set 
ind = int(0.7 * num)
X_test = X[ind : , :, :];
y_test = y[ind : , :];
X = X[: ind, :, :];
y = y[: ind, :];
# build the model: 2 stacked LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(512, return_sequences=True, input_shape=(maxlen, len(chars))))
model.add(Dropout(0.5))
model.add(LSTM(512, return_sequences=False))
model.add(Dropout(0.5))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer=nn_params['opt'])

def sample(a, temperature=1.0):
    # helper function to sample an index from a probability array
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))

# train the model, output generated text after each iteration
max_iter = nn_params['max_iter'] # Maximum number of iterations
lossHist = []
testlossHist = []
trainTime = [0]
testTime = [0]
iterHist = list(range(1,max_iter))
for iteration in range(1, max_iter):
    print()
    print('-' * 50)
    print('Iteration #', iteration)
    #training model
    start_time = time.time()
    print('now it is training...')
    hist = model.fit(X, y, batch_size=128, nb_epoch=1)
    end_time = time.time()
    print('it costs', end_time - start_time, ' for training...')
    trainTime.append(trainTime[-1] + end_time - start_time)
    #testing model
    print('now it is testing...')
    start_time = time.time()
    score = model.evaluate(X_test, y_test, batch_size=128)
    end_time = time.time()
    testTime.append(testTime[-1] + end_time - start_time)
    print('it costs', end_time - start_time, ' for testing...')
    print('The testing error is', score)

    testlossHist.append(score) 
    lossHist = lossHist + hist.history['loss']

    generated = '\n\n\n###iter:' + str(iteration) + '\n'
    sentence = seed
    generated += sentence
    print('----- Generating with seed: "' + sentence + '"')
    
    for i in range(nn_params['len_gen']):
        x = np.zeros((1, maxlen, len(chars)))
        for t, char in enumerate(sentence):
            x[0, t, char_indices[char]] = 1.
    
        preds = model.predict(x, verbose=0)
        next_index = sample(preds[0])
        next_char = indices_char[next_index]
    
        generated += next_char
        sentence = sentence[1:] + next_char
    
    print()
    with open(nn_params['generatefile'], "a") as text_file:
                text_file.write("{0}".format(generated))

#plt.plot(iterHist, lossHist)
#plt.ylabel('Loss')
#plt.xlabel('Iterations')
#plt.savefig('lossPlot.png')
model.save_weights(nn_params['weightfile'])

trainTime = trainTime[1:]
testTime = testTime[1:]
with open(nn_params['lossfile'], 'w') as f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerows(zip(lossHist, trainTime, testlossHist, testTime))
