
seq_length = 25
iterations_train = 200
generated_chars = 100
generated_chars_book = 1000
learning_rate = 0.001  # Higher lr does not mean it's faster!!
batch_size = 2048
chunk_size = 200000
epochs = 2
shuffle_sentences = True
_use_gpu = True
_multi_gpu = True
step_size = 1
_write_book = False
_load_model = False
encoding = "utf-8-sig"  # or utf-8


import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
handler = logging.FileHandler('logger.log', encoding=encoding)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.info('-' * 20 + 'START OF RUN' + '-' * 20)


# -*- coding: utf-8 -*-
logger.info('Loading modules')
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # Stops tf allocating all memory
session = tf.Session(config=config)
print('Stopped tf of using all memory on gpu (only allocates whats needed)')

# Sets keras backend to tensorflow
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'


import sys
import random
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import Reshape
from keras.layers import Convolution1D
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.utils import np_utils
from keras.optimizers import SGD, Adam, RMSprop
from keras.models import load_model
import pdb
import random
import time
import matplotlib
import glob
# Needed to change plot position while calculating. NEEDS TO ADDED BEFORE pyplot import
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


if not _use_gpu:
    """Use gpu if you have many parameters in your model"""
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    print('Using cpu...')
else:
    print('Using gpu...')




os.chdir(os.path.dirname(os.path.abspath(__file__)))  # chdir(foldername(script_location))
os.chdir('books')
# load ascii text and covert to lowercase
raw_text = ''
for filename in os.listdir():
    with open(filename, 'r', encoding=encoding, errors='ignore') as f: # Ignore crappy chars
        raw_text += f.read()
os.chdir('..')



# Lower test data size
#raw_text = raw_text[:int(1e6)]
# Remove
#raw_text = raw_text.lower()  # Remove lower case
#raw_text = raw_text.replace('\n', ' ')  # Fix so that new line does not matter
raw_text = raw_text.replace('  ', '')  # Remove double spaces
raw_text = raw_text.replace('\t', '')  # Remove tabs
#raw_text = ' '.join(raw_text.split())  # Remove double spaces



# create mapping of unique chars to integers, and a reverse mapping
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))
logger.info(char_to_int)

t = time.time()
sentences = []
next_chars = []
for i in range(0, len(raw_text) - seq_length, step_size):
    sentences.append(raw_text[i: i + seq_length])
    next_chars.append(raw_text[i + seq_length])


if shuffle_sentences:
    # Shuffle data
    z = list(zip(sentences, next_chars))
    # => [('Spears', 1), ('Adele', 2), ('NDubz', 3), ('Nicole', 4), ('Cristina', 5)]
    random.shuffle(z)
    sentences, next_chars = zip(*z)
    z = ""  # Clear from ram

# This needs about 1 gb of ram
# summarize the loaded data
n_chars = len(raw_text)
n_vocab = len(chars)
n_patterns = len(sentences)
logger.info(f'It took {round((time.time()-t),2)} seconds to load {n_patterns} patterns')
logger.info("Total Characters: " + str(n_chars))
logger.info("Total Vocab: " + str(n_vocab))
logger.info("Total Words: " + str(len(raw_text.split())))
logger.info("Total Unique Words: " + str(len(set(raw_text.split()))))
logger.info("Total Patterns: " + str(n_patterns))

"""
#Problem: Way to large ram usage
print('Vectorization...')
X = np.zeros((len(sentences), seq_length, len(chars)))
Y = np.zeros((len(sentences), len(chars)))
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_to_int[char]] = 1
    Y[i, char_to_int[next_chars[i]]] = 1

# X is [samples, time steps, features]
#logger.info('Dataset is ' + str(X.nbytes/1e6) + ' mb in memory')

"""

"""
print('Shuffling data...')  # To shuffle or not to shuffle... that is the question
np.random.seed(1) # fix random seed for reproducing results
# SHUFFLE, without it, ROC and AUC doesnt work, bad results etc
s = np.arange(X.shape[0])
np.random.shuffle(s)
X = X[s]
Y = Y[s]
"""

"""
#one hot encode the output variable
#Y = np_utils.to_categorical(Y)  # Old code
#print('X: ' + str(X.shape))
#print('Y: ' + str(Y.shape))

"""


# define the LSTM model
model = Sequential()
model.add(LSTM(1024, input_shape=(seq_length, len(chars)), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(512, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(1024))
#model.add(Dropout(0.0))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))# Try with TimeDistributed
adam = Adam(lr=learning_rate) # lr 0.001 --> default adam
sgd = SGD(lr=learning_rate, momentum=0.9, nesterov=True) # lr 0.001 --> default adam
rmsprop = RMSprop(lr=learning_rate)
model.save('models/current_model.hdf5')  # Save model as template, weights are updated during runtime


# if multiple gpus, change before compile
if _multi_gpu:
    from tensorflow.python.client import device_lib
    devices = device_lib.list_local_devices()
    gpu_count = 0
    for device in devices:
        if device.device_type == "GPU":
            gpu_count += 1
    print(f"Found {gpu_count} usable gpus")
    from keras.utils.training_utils import multi_gpu_model
    model = multi_gpu_model(model, gpus=gpu_count)

# Compile model
model.compile(loss='categorical_crossentropy', optimizer=rmsprop, metrics = ['accuracy'])

# The only way (easy way) to log model summary
def log_func(txt):
    logger.info(txt)
logger.info(model.summary(print_fn=log_func))



# define the checkpoint
if _load_model:
    try:
        model = load_model('models/current_model.hdf5')
        model.load_weights('models/current_weights.h5')
    except:
        print('Something went wrong. No model loaded')


def sample(preds, temperature=1.0):
    """From what I have gathered, the code makes the output more "variable", since if for an example if you get
    Ha, the model might always assume it's Harry while the true word is Hagrid.
    Low temp = conservative, High temp = more creative
    helper function to sample an index from a probability array"""
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds + 1e-8) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def write_a_book():
    book_file = open('generated_book.txt', 'w', encoding=encoding)
    start_index = random.randint(0, len(raw_text) - seq_length - 1)
    sentence = raw_text[start_index: start_index + seq_length]
    target_book = raw_text[start_index + seq_length: start_index + seq_length + generated_chars_book]
    book_file.write("Seed: \n" + sentence)
    print("Seed: \n" + sentence)
    book_file.write("\n" + '-' * 50 + '\n')
    book_file.write("\nTarget: \n" + target_book)  # When not writing log file, no \n is applied, so \n at start
    print("Target: \n" + target_book)
    book_file.write("\n" + '-' * 50 + '\n')
    for temp in [0.01, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]:
        generated_text = ''
        # region generate characters
        for i in range(5000):
            x_pred = np.zeros((1, seq_length, len(chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_to_int[char]] = 1.
            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, temp)
            next_char = int_to_char[next_index]
            generated_text += next_char
            sentence = sentence[1:] + next_char
            print(f'Generated {i} out of 5000 characters', end='\r')
        print('\t\t\t\t\t', end='\r') # Prev print is too long
        book_file.write(f'\nTemperature: {temp}, Generated text: \n' + generated_text)
        book_file.write("\n" + '-' * 50 + '\n')
        print(f'Temperature: {temp}, Generated text: \n' + generated_text)
    book_file.close()
    quit()


if _write_book:
    write_a_book()

plt.figure()
plt.grid()
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('iteration')
plt.legend(['train', 'test'], loc='upper left')
plt.show(block=False)
history_loss_save = []
history_val_loss_save = []

#model.load_weights('models/weights-improvement-00-0.0483.hdf5')
filepath = "models/weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='min')
tensorboard = TensorBoard(log_dir='./tensorboard', histogram_freq=1,
                          write_graph=True, write_images=False)

if not _load_model:
    # Clear previous tensorboards
    files = glob.glob('tensorboard/*')
    for f in files:
        os.remove(f)

# callbacks_list = [checkpoint]
callbacks_list = [tensorboard]


sentences_chunks = []
next_chars_chunks = []
for i in range(0, len(sentences), chunk_size): # 200 000 sentence chunks
    if i == 0:
        sentences_chunks.append(sentences[i:i+chunk_size])
        next_chars_chunks.append(next_chars[i:i+chunk_size])
    else:
        sentences_chunks.append(sentences[i-seq_length:i + chunk_size])
        next_chars_chunks.append(next_chars[i-seq_length:i + chunk_size])

chunk_loop = 0
for iteration in range(iterations_train):

    X = np.zeros((len(sentences_chunks[chunk_loop]), seq_length, len(chars))) # X is [samples, time steps, features]
    Y = np.zeros((len(sentences_chunks[chunk_loop]), len(chars))) # Y is [samples, features]
    for i, sentence in enumerate(sentences_chunks[chunk_loop]):
        for t, char in enumerate(sentence):
            X[i, t, char_to_int[char]] = 1
        Y[i, char_to_int[next_chars_chunks[chunk_loop][i]]] = 1
        print(f'Reshapening  {i} out of {len(sentences_chunks[chunk_loop])} sentences', end='\r')
    chunk_loop += 1
    if chunk_loop > len(sentences_chunks)-1:
        chunk_loop = 0

    logger.info('-' * 10 + f'Iteration {iteration+1} out of {iterations_train}' + '-' * 10 )
    history = model.fit(X, Y, epochs=epochs, batch_size=batch_size,
                        validation_split=0.1, callbacks=callbacks_list,
                        shuffle=True)
    model.save_weights('models/current_weights.h5', overwrite=True)  # Multigpu, cannot save model but only weights
    logger.info('Loss: ' + str(round(history.history['loss'][-1], 4))
                + ' Acc: ' + str(round(history.history['acc'][-1], 4))
                + ' Val Loss: ' + str(round(history.history['val_loss'][-1], 4))
                + ' Val Acc: ' + str(round(history.history['val_acc'][-1], 4)))
    history_loss_save.append(history.history['loss'][-1])
    history_val_loss_save.append(history.history['val_loss'][-1])
    plt.plot(history_loss_save, 'r-')
    plt.plot(history_val_loss_save, 'b-')
    plt.legend(['train', 'test'], loc='upper left')
    plt.pause(0.0000001)

    # Write out how it looks
    start_index = random.randint(0, len(raw_text) - seq_length - 1) # pick a random seed
    sentence = raw_text[start_index: start_index + seq_length]
    target = raw_text[start_index + seq_length: start_index + seq_length + generated_chars]
    logger.info("Seed: \n" + sentence )
    logger.info("Target: \n" + target)
    for temp in [0.01, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]:
        generated_text = ''
        for i in range(generated_chars):
            x_pred = np.zeros((1, seq_length, len(chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_to_int[char]] = 1.
            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, temp)
            next_char = int_to_char[next_index]
            generated_text += next_char
            sentence = sentence[1:] + next_char
            #print(f'Generated {i} out of {generated_chars} characters', end='\r')
        #print('\t\t\t\t\t', end='\r') # Prev print is too long
        logger.info(f'Temperature: {temp}, Generated text: \n' + generated_text)


print('Done')
plt.savefig('train.png')
model.save('Finished_model.hdf5')
