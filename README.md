# -Neural-Network-for-Predicting-Electrical-Power-Consumption-ANNs

### README.md

```markdown
# Neural Network for Predicting Electrical Power Consumption

This project implements a neural network model for predicting electrical power consumption based on multiple features such as temperature, pressure, humidity, and voltage. The model is built using TensorFlow and Keras and performs regression on the electrical power consumption dataset.

## Overview

The script performs the following tasks:
1. **Data Preprocessing**: It loads the electrical power consumption data, splits it into training and testing sets, and normalizes the input features.
2. **Model Construction**: It builds a neural network with three hidden layers using LeakyReLU activation and dropout for regularization.
3. **Model Training**: The neural network is trained on the dataset with validation and saving the trained model.
4. **Model Evaluation**: After training, the model is evaluated on the test set, and predictions are made.
5. **Visualization**: The script visualizes the training loss and mean absolute error (MAE) for both training and validation data.

## Features

- **Input Features**: 
    - `AT`: Ambient Temperature (°C)
    - `V`: Voltage (V)
    - `AP`: Atmospheric Pressure (hPa)
    - `RH`: Relative Humidity (%)
- **Target Variable**: 
    - `PE`: Electrical Power Consumption (kW)

## Requirements

To run the project, ensure you have the following libraries installed:

```bash
pip install tensorflow pandas scikit-learn matplotlib
```

## Running the Script

To train the model, use the following command:

```bash
python neural_network.py --data <path_to_data_file>
```

The default path to the dataset is `electrical_power.csv`, but you can specify a different file path with the `--data` argument.

### Example:

```bash
python neural_network.py --data electrical_power.csv
```

## Dataset Format

The dataset should be a CSV file with the following columns:
- `AT`: Ambient temperature (°C)
- `V`: Voltage (V)
- `AP`: Atmospheric pressure (hPa)
- `RH`: Relative humidity (%)
- `PE`: Electrical power consumption (kW)

### Example of the first few rows:

```csv
AT,V,AP,RH,PE
8.34,40.77,1010.84,90.01,480.48
23.64,58.49,1011.4,74.2,445.75
29.74,56.9,1007.15,41.91,438.76
19.07,49.69,1007.22,76.79,453.09
11.8,40.66,1017.13,97.2,464.43
```

## Model Details

The neural network model architecture is as follows:
- **Input Layer**: 4 input features (`AT`, `V`, `AP`, `RH`)
- **Hidden Layer 1**: 128 neurons, LeakyReLU activation, 30% dropout
- **Hidden Layer 2**: 64 neurons, LeakyReLU activation, 30% dropout
- **Hidden Layer 3**: 32 neurons, LeakyReLU activation
- **Output Layer**: 1 neuron (regression task), no activation function

### Optimizer:
- **Adam optimizer** with default learning rate.

### Loss Function:
- **Mean Squared Error (MSE)** for regression.

### Evaluation Metrics:
- **Mean Absolute Error (MAE)**

## Results

After training, the model is saved to the file `trained_model.h5`. You can use this model for predictions on new data.

### Example of Actual vs Predicted Values:

```text
Actual   Predicted
-------------------
480.48   475.34
445.75   440.60
438.76   432.10
453.09   448.50
464.43   460.20
```

## Visualizations

The following plots are displayed after training:
1. **Training vs Validation Loss**: To visualize how the model's loss evolves over epochs.
2. **Training vs Validation MAE**: To observe the model's performance on the validation set over time.

## Saving and Loading the Model

The trained model is saved in `trained_model.h5`. To load the model later, use the following code:

```python
from tensorflow.keras.models import load_model
model = load_model('trained_model.h5')
```
