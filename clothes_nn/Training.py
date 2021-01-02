import os
import LibAccess
import numpy as np

from lib.tools.Data_Operations import Loading_Operations, Useful_Operations
from lib.Models import Model
from lib.Layers import Layer_Dense
from lib.Activations import Activation_ReLU, Activation_Softmax
from lib.Losses import Loss_CategoricalCrossentropy
from lib.Optimizers import Optimizer_Adam
from lib.Accuracies import Accuracy_Categorical

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

model = Model()
model.add(Layer_Dense(X.shape[1], 128))
model.add(Activation_ReLU())
model.add(Layer_Dense(128, 128))
model.add(Activation_ReLU())
model.add(Layer_Dense(128, 10))
model.add(Activation_Softmax())

model.set(
    loss=Loss_CategoricalCrossentropy(),
    optimizer=Optimizer_Adam(decay=1e-3),
    accuracy=Accuracy_Categorical(),
)

model.finalize()
model.train(
    X, y, validation_data=(X_test, y_test), epochs=10, batch_size=128, print_every=100
)
