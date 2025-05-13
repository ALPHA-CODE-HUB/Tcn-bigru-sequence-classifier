TCN-BiGRU Malware Detection Model
This repository contains the implementation of a hybrid Temporal Convolutional Network and Bidirectional GRU (TCN-BiGRU) model for malware detection using API call sequences.
Overview
This project implements a deep learning approach for malware detection by analyzing sequences of API calls. The model combines the strengths of Temporal Convolutional Networks (TCN) for capturing temporal patterns and Bidirectional Gated Recurrent Units (BiGRU) for processing sequential information in both forward and backward directions.
Dataset
The model uses the MalBehavD-V1 dataset, which contains:

API call sequences extracted from program executions
Binary labels indicating whether a sample is benign (0) or malware (1)
Each sample is represented by its API calls, which are preprocessed and encoded for model input

Model Architecture
The TCN-BiGRU architecture consists of:

Embedding Layer: Transforms encoded API calls into dense vector representations
Batch Normalization: Normalizes the embeddings to improve training stability
Temporal Convolutional Network (TCN): Processes the temporal patterns with dilated convolutions

64 filters
Kernel size of 9
Dilations of [1, 2, 4, 8]
ReLU activation


Bidirectional GRU: Processes the sequence in both directions

256 units
30% dropout for regularization


Dense Layer: Single-unit output layer with sigmoid activation for binary classification

Performance
The model achieves strong performance on malware detection:

Training Accuracy: Maximum of over 99%
Validation Accuracy: Maximum of over 98%
ROC-AUC: Over 0.99, indicating excellent discrimination ability
Comprehensive Evaluation: Full classification metrics including precision, recall, and F1-score
Visualizations: Includes confusion matrix, ROC curve, and precision-recall curve

Installation and Setup
Prerequisites

Python 3.6+
TensorFlow 2.x
Keras
Keras-TCN
NumPy
Pandas
Scikit-learn
Matplotlib
Seaborn

Installation
bash# Clone the repository
git clone https://github.com/yourusername/tcn-bigru-malware-detection.git
cd tcn-bigru-malware-detection

# Install dependencies
pip install -r requirements.txt

# Install the keras-tcn package
pip install keras-tcn
Fix for keras-tcn
The code includes a patch for a compatibility issue in the keras-tcn library:
bashsed -i 's/self.build_output_shape.as_list()/list(self.build_output_shape)/g' /path/to/python/dist-packages/tcn/tcn.py
Usage
Data Preparation
python# Load and preprocess data
data = pd.read_csv('MalBehavD-V1-dataset.csv')
data = data.drop(columns=['sha256'])
y = data['labels']
api_calls = data.drop(columns=['labels']).fillna('')
api_calls = api_calls.apply(lambda x: ' '.join(x.dropna()), axis=1)

# Encode API calls
unique_calls = pd.Series(api_calls.str.split().sum()).unique()
label_encoder = LabelEncoder().fit(unique_calls)
api_calls_encoded = api_calls.apply(lambda x: label_encoder.transform(x.split()))

# Pad sequences
max_sequence_length = 150
X = pad_sequences(api_calls_encoded, maxlen=max_sequence_length, padding='post')
Model Training
python# Create and train the model
model = Sequential(name="TCN-BiGRU_model")
model.add(Embedding(input_dim=unique_api_calls, output_dim=embedding_dim, input_length=max_sequence_length))
model.add(BatchNormalization())
model.add(TCN(nb_filters=64, kernel_size=9, dilations=[1, 2, 4, 8], activation='relu', return_sequences=True))
model.add(Bidirectional(GRU(units=256, return_sequences=False, dropout=0.3)))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(learning_rate=0.0001),
    metrics=['accuracy']
)

history = model.fit(
    X_train, y_train,
    validation_split=0.3,
    epochs=80,
    batch_size=512,
    callbacks=[early_stopping]
)
Evaluation
python# Make predictions and evaluate
y_pred = model.predict(X_test)
y_pred_binary = (y_pred > 0.5).astype(int)

# Generate classification report
print(classification_report(np.array(y_test), y_pred_binary.flatten()))

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred_binary)
Results
The model demonstrates strong performance in malware detection with:

High precision and recall for both benign and malware classes
Minimal false positives and false negatives
Excellent AUC score in ROC curve analysis
Strong precision-recall tradeoff

Visualizations
The repository includes code to generate:

Training and validation curves (loss and accuracy)
Confusion matrix
ROC curve with AUC score
Precision-Recall curve

Citation
If you use this code for your research, please cite:
@article{TCN-BiGRU-Malware,
  title={TCN-BiGRU: A Hybrid Deep Learning Approach for Malware Detection Using API Call Sequences},
  author={Your Name},
  journal={arXiv preprint},
  year={2025}
}
License
This project is licensed under the MIT License - see the LICENSE file for details.
Acknowledgments

MalBehavD-V1 dataset creators
The keras-tcn library maintainers
TensorFlow and Keras development teams
