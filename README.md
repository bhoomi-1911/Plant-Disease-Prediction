Plant Disease Prediction

This project implements a deep learning model using Convolutional Neural Networks (CNNs) in TensorFlow/Keras to predict plant diseases from leaf images. The model is trained on the New Plant Diseases Dataset (Augmented) and classifies plant leaves into 38 different categories (healthy or diseased).

1. Project Overview

A CNN model with ~7.8M parameters was built from scratch.
Trained on augmented plant disease dataset with over 87,000 training images.
Achieved high accuracy in distinguishing between multiple crop diseases.
Model was saved (my_model.h5) and later reloaded for testing and single-image prediction.

2. Dataset

Source: New Plant Diseases Dataset (Augmented)
Classes: 38 (covering apples, grapes, potatoes, tomatoes, corn, etc., with multiple diseases + healthy cases)

Split:
Training set
Validation set
Test set

3. Model Architecture

The CNN consists of stacked convolutional layers with ReLU activation, max pooling, dropout, and dense layers.
Architecture Highlights:
5 Convolutional Blocks (32 → 512 filters)

Dropout Layers (0.25 and 0.4) to reduce overfitting
Dense Layer with 1500 neurons
Output Layer: 38 units with Softmax activation
Total Parameters: 7,842,762 (~30 MB)

4. Training Details

Optimizer: Adam (lr=0.0001)
Loss Function: Categorical Crossentropy
Batch Size: 32
Epochs: 10

Training Metrics
Epoch	TrainingAccuracy	ValidationAccuracy	TrainingLoss	ValidationLoss
  1        41.5%	            83.5%	            2.07	         0.52
  5      	 95.3%            	94.6%	            0.14	         0.17
  10	     98.2%	            97.5%	            0.05         	 0.08

Final Results:
Training Accuracy: 99.6%
Validation Accuracy: 97.5%
Validation Loss: 0.08

5. Model Evaluation
   
Classification Report (Validation Set, 17,572 images)
Overall Accuracy: 98%
Macro Avg (Precision/Recall/F1): 0.98
Weighted Avg (Precision/Recall/F1): 0.98

Sample results:

Apple Scab: Precision 0.96, Recall 0.96
Tomato Yellow Leaf Curl Virus: Precision 0.98, Recall 0.99
Corn Common Rust: Precision 0.99, Recall 0.99
Potato Early Blight: Precision 1.00, Recall 0.99

6. Visualization
Accuracy & Loss Curves
The training and validation curves show steady improvement with minimal overfitting.


Confusion Matrix
A confusion matrix was generated to visualize class-wise performance. Most classes achieved near-perfect classification.
