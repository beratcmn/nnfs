import pandas as pd
import numpy as np


# Set the random seed for reproducibility
np.random.seed(36)

# Load the data from CSV
data = pd.read_csv("./data/binary_500.csv")
X = data[["Feature1", "Feature2"]].values
Y = data["Label"].values

# print(data.head())


# Network architecture
input_size = 2
hidden_size = 3
output_size = 1

# Initialize weights and biases
W1 = np.random.randn(input_size, hidden_size)
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size)
b2 = np.zeros((1, output_size))


# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Derivative of the sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)


def forward_propagation(X):
    # Input to hidden layer
    Z1 = np.dot(X, W1) + b1
    A1 = sigmoid(Z1)

    # Hidden to output layer
    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)

    return A1, A2


def backpropagation(X, Y, A1, A2, learning_rate=0.1):
    global W1, b1, W2, b2

    # Calculate the error
    error = Y - A2
    dA2 = error * sigmoid_derivative(A2)

    # Calculate gradients for W2 and b2
    dW2 = np.dot(A1.T, dA2)
    db2 = np.sum(dA2, axis=0, keepdims=True)

    # Calculate gradients for W1 and b1
    dA1 = np.dot(dA2, W2.T) * sigmoid_derivative(A1)
    dW1 = np.dot(X.T, dA1)
    db1 = np.sum(dA1, axis=0, keepdims=True)

    # Update weights and biases
    W1 += learning_rate * dW1
    b1 += learning_rate * db1
    W2 += learning_rate * dW2
    b2 += learning_rate * db2


def train(X, Y, epochs=1000, learning_rate=0.1):
    for epoch in range(epochs):
        A1, A2 = forward_propagation(X)
        backpropagation(X, Y, A1, A2, learning_rate)

        if epoch % 100 == 0:
            # Calculate the loss (mean squared error)
            loss = np.mean((Y - A2) ** 2)
            print(f"Epoch {epoch}, Loss: {loss:.4f}")


# Convert labels to a 2D array
Y = Y.reshape(-1, 1)

# Train the neural network
train(X, Y, epochs=1000, learning_rate=0.1)


def predict(X):
    _, A2 = forward_propagation(X)
    return (A2 > 0.5).astype(int)


# Make predictions on the training data
predictions = predict(X)

# Calculate accuracy
accuracy = np.mean(predictions == Y)
print(f"Accuracy: {accuracy:.4f}")
