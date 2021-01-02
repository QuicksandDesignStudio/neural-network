import os
import LibAccess
import numpy as np

from lib.tools.Data_Operations import Loading_Operations, Useful_Operations

load_data = Loading_Operations()
data_operations = Useful_Operations()

X, y, X_test, y_test = load_data.create_folder_labelled_data(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "fashion_mnist_images")
)

shuffled_keys = data_operations.shuffle_keys(X)
X = X[shuffled_keys]
y = y[shuffled_keys]


X = data_operations.linear_scaling_negative1_positive1(X, 127.5, np.float32)
X_test = data_operations.linear_scaling_negative1_positive1(X_test, 127.5, np.float32)


X = X.reshape(X.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)
