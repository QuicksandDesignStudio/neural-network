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

X, y, X_test, y_test = Loading_Operations.create_folder_labelled_data(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "fashion_mnist_images")
)


shuffled_keys = Useful_Operations.shuffle_keys(X)
X = X[shuffled_keys]
y = y[shuffled_keys]


X = Useful_Operations.linear_scaling_negative1_positive1(X, 127.5, np.float32)
X_test = Useful_Operations.linear_scaling_negative1_positive1(X_test, 127.5, np.float32)


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


"""
# load and evaluate model example
model.save("fashion_minst.model")
model = Model.load("fashion_minst.model")
model.evaluate(X_test, y_test)
"""

"""
#save and load params example
parameters = model.get_parameters()
model.save_parameters("fashion_minst.params")
print("--------------------------------------------------")
# set parameters and evaluate without trainings

model = Model()
model.add(Layer_Dense(X.shape[1], 128))
model.add(Activation_ReLU())
model.add(Layer_Dense(128, 128))
model.add(Activation_ReLU())
model.add(Layer_Dense(128, 10))
model.add(Activation_Softmax())

model.set(loss=Loss_CategoricalCrossentropy(), accuracy=Accuracy_Categorical())
model.finalize()
model.load_parameters("fashion_minst.params")
model.evaluate(X_test, y_test)
"""