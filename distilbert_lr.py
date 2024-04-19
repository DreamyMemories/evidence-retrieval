from transformers import DistilBertTokenizer, TFDistilBertModel
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib
# import tensorflow as tf

# # check gpu
# physical_devices = tf.config.list_physical_devices('GPU')
# print("Num GPUs Available: ", len(physical_devices))
def get_distilbert_embeddings(data, tokenizer, model, batch_size=32):
    # Placeholder for the embeddings
    all_embeddings = []

    for i in range(0, len(data), batch_size):
        batch = data.iloc[i:i+batch_size]
        # Combine claim and evidence into one string per pair
        texts = list(batch['Claim'] + " [SEP] " + batch['Evidence'])
        inputs = tokenizer.batch_encode_plus(texts, padding='max_length', truncation=True, return_tensors="tf", max_length=110)

        # Generate embeddings
        outputs = model(inputs['input_ids'], attention_mask=inputs['attention_mask'])
        # Use the last_hidden_state so compatible with LSTM
        embeddings = outputs.last_hidden_state.numpy()
        all_embeddings.append(embeddings)
        

    # Concatenate all batch embeddings into a single array
    # print(np.array(all_embeddings).shape)
    return np.vstack(all_embeddings)


# Read data
train_data = pd.read_csv('./train.csv')
validation_data = pd.read_csv('./dev.csv')

# Get the labels
train_labels = train_data['label'].values
validation_labels = validation_data['label'].values

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = TFDistilBertModel.from_pretrained('distilbert-base-uncased')

# Generate embeddings for your data
training_embeddings = get_distilbert_embeddings(train_data, tokenizer, model)
validation_embeddings = get_distilbert_embeddings(validation_data, tokenizer, model)

# The task is a pairwise sequence classification problem. use traditional 

# Flatten the embeddings
training_embeddings_flat = training_embeddings.reshape(training_embeddings.shape[0], -1)
validation_embeddings_flat = validation_embeddings.reshape(validation_embeddings.shape[0], -1)


# Initialize the Logistic Regression model
lr_model = LogisticRegression(max_iter=1000)

# Train the model
lr_model.fit(training_embeddings_flat, train_labels)

# save the model
joblib.dump(lr_model, 'lr_model.pkl')

# Predict on the validation set
validation_predictions = lr_model.predict(validation_embeddings_flat)

# get the accuracy
print("Accuracy: ", np.mean(validation_predictions == validation_labels))

# Evaluate the model
print(classification_report(validation_labels, validation_predictions))