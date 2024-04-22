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

This model is a classical approach using logistic regression that was trained
      on 23K pairs of texts.

- **Developed by:** Wei Xiang Wong and Enlong Bo
- **Language(s):** English
- **Model type:** Supervised
- **Model architecture:** Logistic Regression
- **Finetuned from model [optional]:** N/A

### Model Resources

<!-- Provide links where applicable. -->

- **Repository:** https://github.com/DreamyMemories/evidence-retrieval/tree/main/models/lr_model.pkl 
- **Paper or documentation:** N/A

## Training Details

### Training Data

<!-- This is a short stub of information on the training data that was used, and documentation related to data pre-processing or additional filtering (if applicable). -->

23K pairs of evidence and claim turned into word embeddings where the 'Evidence' and 'Claim' are concatenated together with a [SEP] token using DistilBert. Initial preprocessing also included the removal of `[REF]` tokens from the `Evidence` column.

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

#### Training Hyperparameters

<!-- This is a summary of the values of hyperparameters used in training the model. -->


      - Max Iteration: 1000

#### Speeds, Sizes, Times

<!-- This section provides information about how roughly how long it takes to train the model and the size of the resulting model. -->


      - overall training time: 10 seconds
      - model size: 661KB

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

The model obtained an accuracy of 82%, a recall of 63%, precision of 68% and F1-score of 65%.

## Technical Specifications

### Hardware


      - RAM: at least 16 GB
      - Storage: at least 100MB,
      - GPU: RTX 2060

### Software


      - Transformers 4.40.0
      - Scikit-learn 1.0.2
      - Numpy 1.20.1
      - Pandas 1.2.3 
      - joblib 1.1.0 

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

Any inputs (concatenation of two sequences) longer than
      110 tokens will be truncated by the model. This is due to the limitation of hardware where the memory is not enough to handle the word embeddings.

## Additional Information

<!-- Any other information that would be useful for other people to know. -->

110 tokens was the sweet spot we found where the model was able to handle the word embeddings without running out of memory on our hardware while maximising possible performance.
