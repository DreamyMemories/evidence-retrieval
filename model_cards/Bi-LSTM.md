---
{}
---
language: en
license: cc-by-4.0
tags:
- text-classification
repo: https://github.com/DreamyMemories/evidence-retrieval

---

# Model Card for d38971ww-t75255eb-ED

<!-- Provide a quick summary of what the model is/does. -->

This is a classification model that was trained to
      detect whether a claim is true given the evidence.


## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->

This model is based upon a BERT model that was fine-tuned
      on 30K pairs of texts.

- **Developed by:** Wei Xiang Wong and Enlong Bo
- **Language(s):** English
- **Model type:** Supervised
- **Model architecture:** Bi-LSTM
- **Finetuned from model [optional]:** N/A

### Model Resources

<!-- Provide links where applicable. -->

- **Repository:** https://github.com/DreamyMemories/evidence-retrieval/tree/main/models/distilbert_lstm2
- **Paper or documentation:** N/A

## Training Details

### Training Data

<!-- This is a short stub of information on the training data that was used, and documentation related to data pre-processing or additional filtering (if applicable). -->

23K pairs of evidence and claim which are augmented once through insertion further explained in "Additional Information"  which are then turned into word embeddings using DistilBert.

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

#### Training Hyperparameters

<!-- This is a summary of the values of hyperparameters used in training the model. -->


      - learning_rate: 0.001
      - train_batch_size: 64
      - eval_batch_size: 64
      - num_epochs: 20

#### Speeds, Sizes, Times

<!-- This section provides information about how roughly how long it takes to train the model and the size of the resulting model. -->


      - overall training time: 1 hour
      - duration per training epoch: 3 minutes
      - model size: 12.6MB

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data & Metrics

#### Testing Data

<!-- This should describe any evaluation data used (e.g., the development/validation set provided). -->

The development set provided, amounting to 6K pairs.

#### Metrics

<!-- These are the evaluation metrics being used. -->


      - Precision
      - Recall
      - F1-score
      - Accuracy

### Results

The model obtained an accuracy of 85%, a recall of 71%, precision of 73% and F1-score of 72%.

## Technical Specifications

### Hardware


      - RAM: at least 16 GB
      - Storage: at least 2GB,
      - GPU: V100

### Software


      - Transformers 4.40.0
      - TensorFlow 2.10.1
      - Numpy 1.20.1
      - Pandas 1.2.3 

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

Any inputs (concatenation of two sequences) longer than
      110 subwords will be truncated by the model.

## Additional Information

<!-- Any other information that would be useful for other people to know. -->

Initial data pre processing is done through data augmentation using DistilBert embeddings to replace words that are contextually similar with p_aug = 0.3, aug_min = 1, top_k = 20 using nlaug library.