import pandas as pd
import numpy as np
import tensorflow as tf
import string
import tensorflow_hub as hub

import matplotlib.pyplot as plt

from model import create_model
from preprocessing import preprocess_data, encode_labels, one_hot_labels
from evaluation import calculate_results
### Preprocess


## Assigning paths
path = "PubMed_20k_RCT_numbers_replaced_with_at_sign/"
train_dir = path + "train.txt"
val_dir = path + "dev.txt"
test_dir = path + "test.txt"

## Preprocess samples
train_samples = preprocess_data(train_dir)
val_samples = preprocess_data(val_dir)
test_samples = preprocess_data(test_dir)

## Transfer it to a dataframe
train_df = pd.DataFrame(train_samples)
val_df = pd.DataFrame(val_samples)
test_df = pd.DataFrame(test_samples)

# Sentences
train_sentences = train_df["text"].to_list()
val_sentences = val_df["text"].to_list()
test_sentences = test_df["text"].to_list()

# Labels
train_labels = train_df["target"].to_list()
val_labels = val_df["target"].to_list()
test_labels = test_df["target"].to_list()

# Line number
train_line_number = train_df["line_number"].to_list()
val_line_number = val_df["line_number"].to_list()
test_line_number = test_df["line_number"].to_list()

# Total lines
train_total_lines = train_df["total_lines"].to_list()
val_total_lines = val_df["total_lines"].to_list()
test_total_lines = test_df["total_lines"].to_list()

# ------------------------------------------------


### Encode and One hot labels

# Encode the labels
train_labels_encoded, val_labels_encoded = encode_labels(train_labels, val_labels)

# One hot encode the labels
train_labels_one_hot, val_labels_one_hot =  one_hot_labels(train_df, val_df)

# Shape of line numbers that covers 98 percent of the data
line_number_percentage = int(np.percentile(train_line_number, 98))

train_line_number_one_hot = tf.one_hot(train_df["line_number"].to_numpy(),depth=line_number_percentage)
val_line_number_one_hot = tf.one_hot(val_df["line_number"].to_numpy(),depth=line_number_percentage)

# Shape of total lines that covers 98 percent of the data
total_lines_percentage = int(np.percentile(train_total_lines, 98))

train_total_lines_one_hot = tf.one_hot(train_df["total_lines"].to_numpy(),depth=total_lines_percentage)
val_total_lines_one_hot = tf.one_hot(val_df["total_lines"].to_numpy(),depth=total_lines_percentage)

# ------------------------------------------------

### Char Vectorization, Char embedding and Pretrained token embedding


# Split a sentence into characters
def split_chars(sentence):

  delimiter = " "

  return delimiter.join(list(sentence))


# Get all the lowercase letters, digits and punctuations in order to calculate the number of max tokens in the vectorization
alphabet = string.ascii_lowercase + string.digits + string.punctuation
num_chars = len(alphabet) + 2 # 2 is for space and unknown characters

# Split the sentences into characters
train_chars = [split_chars(sentence) for sentence in train_sentences]
val_chars = [split_chars(sentence) for sentence in val_sentences]
test_chars = [split_chars(sentence) for sentence in test_sentences]

# Number of chars that covers 95 percent of the sentences in the dataset
all_chars = [len(sentence) for sentence in train_sentences]
char_percentage = int(np.percentile(all_chars, 95))

# Creating a char vectorization layer
char_vectorization = tf.keras.layers.TextVectorization(max_tokens=num_chars,
                                                       output_sequence_length=char_percentage)

# Adapt it into the train characters
char_vectorization.adapt(train_chars)

# Creating embedding layer of the characters
char_embedding = tf.keras.layers.Embedding(input_dim=num_chars,
                                           output_dim=25) # from paper

# The paper has used Glove but since it is not in tensorfow hub, I used universal sentence encoder
token_embeddings_layer = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder/4",trainable=False)

# -----------------------------------------------------------

# Create the model
model = create_model(char_embedding, char_vectorization, token_embeddings_layer)

# -----------------------------------------------------------


# Create the datasets using tf.data.Dataset so that the model can work faster
train_features = tf.data.Dataset.from_tensor_slices((train_sentences, train_chars,
                                                    train_line_number_one_hot, train_total_lines_one_hot))

train_labels = tf.data.Dataset.from_tensor_slices(train_labels_one_hot)
train_dataset = tf.data.Dataset.zip((train_features, train_labels))

val_features = tf.data.Dataset.from_tensor_slices((val_sentences, val_chars, val_line_number_one_hot, val_total_lines_one_hot))
val_labels = tf.data.Dataset.from_tensor_slices(val_labels_one_hot)
val_dataset = tf.data.Dataset.zip((val_features, val_labels))

train_dataset = train_dataset.batch(32).prefetch(tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(32).prefetch(tf.data.AUTOTUNE)

# Fit the model
history = model.fit(train_dataset, validation_data=val_dataset, epochs=5)

# -----------------------------------------------------------

# Evaluate the model
model_preds_probs = model.predict(val_dataset)
model_preds = tf.argmax(model_preds_probs, axis=1)

model_results = calculate_results(val_labels_encoded, model_preds)
model_results = model_results.transpose()

model_results.plot(kind="bar",figsize=(10,7))
plt.show()





