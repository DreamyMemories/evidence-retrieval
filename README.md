# Evidence Retrieval

This repository contains code and models for a text classification project using the DistilBERT architecture. The models are trained to determine given a claim and evidence whether it is true or false. This README provides an overview of the project structure, setup instructions, and how to run the models for predictions.

## Project Structure

- `distilbert_lr.py`: This script trains a logistic regression model using embeddings from DistilBERT.
- `distilbert_lstm.py`: This script trains an LSTM model using DistilBERT embeddings.
- `inferenceNLU.ipynb`: A Jupyter notebook that loads trained models and performs predictions on new data for live demo purposes.
- `distilbert_lstm2`: A directory containing the trained LSTM model saved in TensorFlow format.
- `lr_model.pkl`: A pickle file containing the trained logistic regression model.

## Setup
To run the scripts and notebooks in this repository, you will need to install the required dependencies. You will need the following packages

- TensorFlow
- Transformers by Hugging Face
- Pandas
- Scikit-learn
- Numpy
- nlpaug

You can install these packages using pip:

```bash
pip install tensorflow transformers pandas scikit-learn numpy nlpaug
```

## Running the Models

To run the models, open up the `prediction.ipynb` notebook in Jupyter and follow the instructions to load the models and perform predictions on new data.

