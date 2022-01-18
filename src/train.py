import random
from typing import Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import (
    Input,
    Embedding,
    SpatialDropout1D,
    LSTM,
    Bidirectional,
    Conv1D,
    GlobalAveragePooling1D,
    GlobalMaxPooling1D,
    Dense,
    concatenate
)

from utils.config import DATA_DIR, DATASET, ROOT_DIR, SEED
from pipeline.preprocess import clean_text

data_path = ROOT_DIR / DATA_DIR / DATASET
dataset = pd.read_csv(data_path, compression=data_path.suffix.strip('.'))

def set_seed(seed: int) -> None:
    """Set random seed for reproducability."""

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    print(f'Seed set to {seed}')

set_seed(SEED)

TARGETS = ['toxic', 
           'severe_toxic', 
           'obscene', 
           'threat',
           'insult', 
           'identity_hate']

train, valid = train_test_split(dataset, test_size=0.2, random_state=SEED)
train, test = train_test_split(train, test_size=0.2, random_state=SEED)

print(f'Train Size: {train.shape}')
print(f'Valid Size: {valid.shape}')
print(f'Test Size: {test.shape}')

train_comments = train.comment_text.values
train_toxicity = train[TARGETS].values

val_comments = valid.comment_text.values
val_toxicity = valid[TARGETS].values

# Save test dataset to root_dir/data.

test_comments = test.comment_text.values
test_toxicity = test[TARGETS].values

vfunc = np.vectorize(clean_text)
train_norm_comments = vfunc(train_comments)
val_norm_comments = vfunc(val_comments)
test_norm_comments = vfunc(test_comments)

MAX_WORD_FREQUENCY = None
OOV_TOKEN = '<UNK>'
PADDING: Union['pre', 'post'] = 'post'
TRUNCATE: Union['pre', 'post'] = 'post'
EMBED_SIZE = 100

tokenizer = Tokenizer(num_words=MAX_WORD_FREQUENCY, oov_token=OOV_TOKEN)
tokenizer.fit_on_texts(train_norm_comments)

train_comment_sequences = tokenizer.texts_to_sequences(train_norm_comments)
val_comment_sequences = tokenizer.texts_to_sequences(val_norm_comments)
test_comment_sequences = tokenizer.texts_to_sequences(test_norm_comments)

MAX_LENGTH = 200
VOCAB_SIZE = len(tokenizer.word_index)

# maybe change dtype to int16 or int8, might add speed
X_train = pad_sequences(train_comment_sequences, maxlen=MAX_LENGTH, padding=PADDING, truncating=TRUNCATE)
X_val = pad_sequences(val_comment_sequences, maxlen=MAX_LENGTH, padding=PADDING, truncating=TRUNCATE)
X_test = pad_sequences(test_comment_sequences, maxlen=MAX_LENGTH, padding=PADDING, truncating=TRUNCATE)

y_train = train_toxicity.copy()
y_val = val_toxicity.copy()
y_test = test_toxicity.copy()

sequence_input = Input(shape=(MAX_LENGTH))
x = Embedding(input_dim=VOCAB_SIZE + 1, output_dim=EMBED_SIZE, mask_zero=True)(sequence_input)
x = SpatialDropout1D(0.35)(x)
x = Bidirectional(LSTM(128, return_sequences=True,dropout=0.15))(x)
x = Bidirectional(LSTM(128, return_sequences=True,dropout=0.1))(x)
x = Conv1D(64, kernel_size = 3, padding = "valid", kernel_initializer = "glorot_uniform")(x)
avg_pool = GlobalAveragePooling1D()(x)
max_pool = GlobalMaxPooling1D()(x)
x = concatenate([avg_pool, max_pool])
preds = Dense(6, activation="sigmoid")(x)

model = Model(inputs=sequence_input, outputs=preds)
model.compile(optimizer=tf.optimizers.Adam(learning_rate=1e-3), 
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=False, axis=0),
              metrics=['accuracy'])

class RoCAUCEvalCallback(tf.keras.callbacks.Callback):
    def __init__(self, validation_data: tuple):
        super().__init__()
        self.X_val, self.y_val = validation_data
        
    def on_epoch_end(self, epoch, logs=None):
        y_pred = self.model.predict(self.X_val)
        score = roc_auc_score(self.y_val, y_pred)
        print(f'RoC AUC Socre: {score}')

checkpoints_path = 'checkpoints/'
best_model = 'best_model/'
EPOCHS = 4

model.fit(
    x = X_train,
    y = y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    callbacks=[RoCAUCEvalCallback((X_val, y_val)), 
               tf.keras.callbacks.ModelCheckpoint(
                   filepath=checkpoints_path,
                   save_weights_only=False
               ),
               tf.keras.callbacks.ModelCheckpoint(
                   filepath=best_model,
                   save_best_only=True,
                   save_weights_only=False
               )]
)
