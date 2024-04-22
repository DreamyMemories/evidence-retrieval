from transformers import DistilBertTokenizer, TFDistilBertModel
import numpy as np
import pandas as pd
from keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, SpatialDropout1D
import tensorflow as tf
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import os
import nlpaug.augmenter.word as naw
os.environ['TF_FORCE_GPU_ALLOW_GROWTH']='true'

# import tensorflow as tf

# # check gpu
# physical_devices = tf.config.list_physical_devices('GPU')
# print("Num GPUs Available: ", len(physical_devices))
def get_distilbert_embeddings(data, tokenizer, model, batch_size=32):
    # Placeholder for the embeddings
    all_embeddings = []

    # Do it batch by batch to avoid memory issues
    for i in range(0, len(data), batch_size):
        batch = data.iloc[i:i+batch_size]
        # Combine claim and evidence into one string per pair
        texts = list(batch['Claim'] + " [SEP] " + batch['Evidence'])
        inputs = tokenizer.batch_encode_plus(texts, padding='max_length', truncation=True, return_tensors="tf", max_length=70)

        # Generate embeddings
        outputs = model(inputs['input_ids'], attention_mask=inputs['attention_mask'])
        # Use the last_hidden_state so compatible with LSTM
        embeddings = outputs.last_hidden_state.numpy()
        all_embeddings.append(embeddings)

    return np.vstack(all_embeddings)

# Contextual word embeddings augmenter using BERT
def augment_data_bert(sentences, num_augments=1):
    aug = naw.ContextualWordEmbsAug(
        model_path='distilbert-base-uncased', action="substitute", aug_p=0.3, aug_min=1, device='cuda', top_k=20)
    augmented_sentences = aug.augment(sentences, n=num_augments)
    return augmented_sentences

# Read Data
train_data = pd.read_csv('./data/train.csv')
validation_data = pd.read_csv('./data/dev.csv')

# Remove the [REF] tokens from the evidence
train_data['Evidence'] = train_data['Evidence'].replace(r'[REF]', '', regex=True)
validation_data['Evidence'] = validation_data['Evidence'].replace(r'[REF]', '', regex=True)

# Get the labels for prediction metrics
train_labels = train_data['label'].values
validation_labels = validation_data['label'].values

# Apply augmentation
augmented_claims_bert = augment_data_bert(train_data['Claim'].tolist())

# Extend original training data with augmented data
augmented_data = pd.DataFrame({
    'Claim': augmented_claims_bert,
    'Evidence': train_data['Evidence'].tolist(),
    'label': train_data['label'].tolist()
})
# Combine original and augmented data
train_data_augmented = pd.concat([train_data, augmented_data], ignore_index=True)

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = TFDistilBertModel.from_pretrained('distilbert-base-uncased')

# Generate embeddings for your data
training_embeddings = get_distilbert_embeddings(train_data, tokenizer, model)
augmented_embeddings = get_distilbert_embeddings(augmented_data, tokenizer, model)
validation_embeddings = get_distilbert_embeddings(validation_data, tokenizer, model)

# store embeddings and labels into dictionary, concatenate train and augmented into one. Shuffle the data and split into half.
train_data = np.concatenate((training_embeddings, augmented_embeddings), axis=0)
train_labels = np.concatenate((train_labels, train_labels), axis=0)

# Shuffle the data
np.random.seed(42)
shuffle_indices = np.random.permutation(len(train_data))
train_data = train_data[shuffle_indices]
train_labels = train_labels[shuffle_indices]

# Split the data into training and validation
split_index = len(train_data) // 2
first_half, second_half = train_data[:split_index], train_data[split_index:]
first_half_labels, second_half_labels = train_labels[:split_index], train_labels[split_index:]

# Embedding dimension
embedding_dim = 768  

# Define the model
model = Sequential()
model.add(SpatialDropout1D(0.3))
model.add(Bidirectional(LSTM(128, activation='tanh', recurrent_activation='sigmoid',
                             dropout=0.3, recurrent_dropout=0.3, input_shape=(None, embedding_dim))))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))  # For binary classification

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=int(len(training_embeddings) // 128),
    decay_rate=0.9,
    staircase=True)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

model.compile(optimizer=Adam(learning_rate=lr_schedule), loss='binary_crossentropy', metrics=['accuracy'])

model.fit(
    first_half, first_half_labels,  # Training data and labels
    batch_size=64,
    epochs=10,
    validation_data=(validation_embeddings, validation_labels),
    callbacks=[early_stopping]
)

model.fit(
    second_half, second_half_labels,  # Training data and labels
    batch_size=64,
    epochs=5,
    validation_data=(validation_embeddings, validation_labels),
    callbacks=[early_stopping]
)

# save in tf format
model.save('distilbert_lstm2', save_format='tf')
