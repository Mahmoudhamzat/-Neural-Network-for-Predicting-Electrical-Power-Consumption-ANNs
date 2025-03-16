# -*- coding: utf-8 -*-
"""
Neural Network for Predicting Electrical Power Consumption

This script trains a neural network model on the electrical power consumption dataset. 
It includes data preprocessing, model training, evaluation, and visualization.

Author: Your Name
Created on: Thu Jan 2, 2025
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU
import argparse
import matplotlib.pyplot as plt

# Argument parser for flexibility in file input
parser = argparse.ArgumentParser(description="Train a neural network on electrical power consumption dataset.")
parser.add_argument("--data", type=str, default="electrical_power.csv", help="Path to the dataset")
args = parser.parse_args()

file_path = args.data

# Check if the dataset exists
if not os.path.exists(file_path):
    raise FileNotFoundError(f"File '{file_path}' not found. Please check the file path.")

# Load the data from the CSV file
data = pd.read_csv(file_path)

# Separate input features (X) and target variable (y)
X = data[['AT', 'V', 'AP', 'RH']].values
y = data['PE'].values

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the data using StandardScaler (mean=0, std=1)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build the neural network model
model = Sequential([
    Dense(128, input_dim=4),  # First hidden layer with 128 neurons
    LeakyReLU(alpha=0.1),  # Apply LeakyReLU activation to handle negative values better
    Dropout(0.3),  # Dropout with 30% to prevent overfitting

    Dense(64),  # Second hidden layer with 64 neurons
    LeakyReLU(alpha=0.1),
    Dropout(0.3),

    Dense(32),  # Third hidden layer with 32 neurons
    LeakyReLU(alpha=0.1),

    Dense(1)  # Output layer with 1 neuron (for regression task)
])

# Compile the model using Adam optimizer and MSE as the loss function
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Start training the model
print("🚀 Starting model training, this may take some time...")

# Train the model with training data, validation split of 20%, and 150 epochs
history = model.fit(X_train, y_train, epochs=150, batch_size=32, validation_split=0.2, verbose=1)

# Save the trained model
model.save("trained_model.h5")
print("✅ Model saved successfully as 'trained_model.h5'")

# Evaluate the model on the test set
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
print(f"Mean Absolute Error on Test Data: {test_mae}")

# Make predictions on the test set
predictions = model.predict(X_test)

# Display some of the actual vs predicted values for verification
print("\n📊 Actual vs Predicted Values:")
print(f"{'Actual':<10}{'Predicted':<10}")
print("-" * 20)
for i in range(5):
    print(f"{y_test[i]:<10.2f}{predictions[i][0]:<10.2f}")

# Plot the model's training and validation performance
plt.figure(figsize=(12, 5))

# Plot the training and validation loss (MSE)
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss (MSE)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plot the training and validation MAE
plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Mean Absolute Error (MAE)')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.legend()

plt.tight_layout()
plt.show()
