import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# prepare data
    # get data
rawdata = datasets.load_breast_cancer()
X, y = rawdata.data, rawdata.target

n_samples, n_features = X.shape

    # split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    # scaler data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

    # cast data to torch tensors
X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

    # make y_train and y_test column vectors
y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)

# Model  and loss

class LogisticRegression(nn.Module):
    def __init__(self, n_in_features):
        super().__init__()
        self.linear = nn.Linear(in_features=n_in_features, out_features=1)

    def forward(self, x):
        y_predict = torch.sigmoid(self.linear(x))
        return y_predict

learning_rate = 0.01
n_epochs = 1000

model = LogisticRegression(n_features)
loss = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Print the weights and bias for the trained model
print(f"Weights before training : {model.linear.weight}")
print(f"Bias before training: {model.linear.bias}")

# Training loop
for epoch in range(n_epochs):
    # forward pass and loss
    y_pred = model(X_train)
    l = loss(y_pred, y_train)

    # evaluation in progress
    accurary = y_pred.round().eq(y_train).sum() / float(y_train.shape[0])

    # backward
    l.backward()
    optimizer.step()

    # zero gradient history
    optimizer.zero_grad()

    # progress
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch + 1}    Loss {l.item():.7f}     Accuracy : {accurary.item():.7f}")

# Final evaluation
with torch.no_grad():
    y_pred = model(X_train)
    accurary_train = y_pred.round().eq(y_train).sum() / float(y_train.shape[0])
    print(f"Accuracy on train data : {accurary.item():.7f}")
    y_pred = model(X_test)
    accurary_test = y_pred.round().eq(y_test).sum() / float(y_test.shape[0])
    print(f"Accuracy on test data : {accurary.item():.7f}")
    # Print the weights and bias for the trained model
    print(f"Weights after training : {model.linear.weight}")
    print(f"Bias after training: {model.linear.bias}")