import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def generate_data(num_samples=100):
    # Generate random points
    np.random.seed(36)  # For reproducibility
    X = np.random.randn(num_samples, 2)

    # Assign labels based on a simple linear separation
    Y = (X[:, 0] + X[:, 1] > 0).astype(int)

    return X, Y


def plot_data(X, Y):
    plt.scatter(X[Y == 0][:, 0], X[Y == 0][:, 1], color="red", label="Class 0")
    plt.scatter(X[Y == 1][:, 0], X[Y == 1][:, 1], color="blue", label="Class 1")
    plt.legend()
    plt.show()


def save_to_csv(X, Y, filename="data.csv"):
    data = np.hstack((X, Y.reshape(-1, 1)))
    df = pd.DataFrame(data, columns=["Feature1", "Feature2", "Label"])
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")


# Generate and plot data
X, Y = generate_data(500)
plot_data(X, Y)
save_to_csv(X, Y, "./data/binary_500.csv")
