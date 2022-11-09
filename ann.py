from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data import preprocessed

DATA_PATH = Path(__file__).parent / 'audit_data' / 'audit_risk.csv'
COLUMNS = ['PARA_A', 'Risk_A', 'PARA_B', 'Risk_B', 'TOTAL', 'Money_Value', 'Risk']
TARGET_COL = 'Risk'
INPUT_COLS = COLUMNS.copy()
INPUT_COLS.remove(TARGET_COL)

def main():
    # Read data fuke
    data = preprocessed(DATA_PATH, COLUMNS)

    # Split data for training and testing
    train_data = data.sample(frac=0.8)
    test_data = data.drop(train_data.index)

    # Instantiate Artificial Neural Network
    ann = ArtificialNeuralNetwork(0.1, len(data.columns)-1)

    # Train Artificial Neural Network
    cumulative_errors = ann.train(train_data, INPUT_COLS, TARGET_COL, 1000)

    # Create graph of the training process
    fig, ax = plt.subplots()
    fig.suptitle("Training process")
    ax.plot([iteration*100 for iteration in range(1, len(cumulative_errors)+1)], cumulative_errors)
    ax.set_xlabel("Number of iterations")
    ax.set_ylabel("Cumulative Errors")
    plt.show()


    # Predicts test data
    correct = 0
    for _, test_vector in test_data.iterrows():
        target = test_vector[TARGET_COL]
        test_vector = test_vector[INPUT_COLS].to_numpy(dtype=np.float64)

        result = 1 if ann.predict(test_vector) > 0.5 else 0
        if result == target:
            correct+=1
    print(f"Correctly predicted {correct} out of {len(test_data)} from test data")



class ArtificialNeuralNetwork:
    def __init__(self, learning_rate, input_num):
        '''
            Instantiate an Artifial Neural Network with set learning rate and number of input
        '''
        # Assigning random weights equal to the number of input
        self.weights = np.array(list(np.random.randn() for _ in range(input_num)))

        self.bias = np.random.randn()
        self.learning_rate = learning_rate

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _sigmoid_derivative(self, x):
        return self._sigmoid(x) * (1 - self._sigmoid(x))

    def predict(self, input_vector):
        '''
            Return prediction value.
            Uses 2 layer.
        '''
        layer1 = np.dot(input_vector, self.weights)
        layer2 = self._sigmoid(layer1)
        prediction = layer2
        return prediction

    def _compute_gradients(self, input_vector, target):
        '''
            Return error derivative of weights and error derivative of bias
        '''
        layer1 = np.dot(input_vector, self.weights)
        layer2 = self._sigmoid(layer1)
        prediction = self.predict(input_vector)

        derror_dprediction = 2 * (prediction - target)
        dprediction_dlayer1 = self._sigmoid_derivative(layer1)

        dlayer1_dbias = 1
        dlayer1_dweights = input_vector

        derror_dbias = derror_dprediction * dprediction_dlayer1 * dlayer1_dbias
        derror_dweights = derror_dprediction * dprediction_dlayer1 * dlayer1_dweights

        return derror_dbias, derror_dweights

    def update_parameters(self, derror_dbias, derror_dweights):
        self.bias -= derror_dbias*self.learning_rate
        self.weights -= derror_dweights*self.learning_rate

    def update_bias(self, derror_dbias):
        self.bias = self.bias - derror_dbias*self.learning_rate

    def update_weights(self, derror_dweights):
        self.weights = self.weights - derror_dweights*self.learning_rate

    def train(self, input_vectors, input_cols, target_col, iterations):
        '''
            Train the ann for a number of iterations.
            Return cumulative errors for every 100 iteration.
        '''
        cumulative_errors = []
        for current_iteration in range(1, iterations+1):
            # Pick a datapoint at random as input vector
            input_vector = input_vectors.sample(replace=True)
            target = input_vector[target_col].iloc[0]
            input_vector = input_vector[input_cols].to_numpy(dtype=np.float64)[0]

            # Compute the gradients
            derror_dbias, derror_dweights = self._compute_gradients(input_vector, target)

            # Update the parameters
            self.update_bias(derror_dbias)
            self.update_weights(derror_dweights)

            # Measure the cumulative error for all the instances
            if current_iteration % 100 == 0:
                cumulative_error = 0
                # Loop through all the instances to measure the error
                for _, input_vector in input_vectors.iterrows():
                    target = input_vector[target_col]
                    input_vector = input_vector[input_cols].to_numpy(dtype=np.float64)

                    prediction = self.predict(input_vector)
                    error = np.square(prediction - target)

                    cumulative_error += error
                cumulative_errors.append(cumulative_error)

        return cumulative_errors

if __name__ == '__main__':
    main()