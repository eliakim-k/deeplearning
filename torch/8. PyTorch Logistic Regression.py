import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Prepare data
    # get data
rawdata = datasets.load_breast_cancer()
X, y = rawdata.data, rawdata.target

    # samples and features
n_samples, n_features = X.shape

    # preprocess data : split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
    # preprocess data : scale data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
    # cast to torch tensors
X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))
    # preprocess data : make y_train and y_test column vectors
y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)

# Model
class LinearRegression(nn.Module):
    def __init__(self, n_input_features):
        super().__init__()
        self.linear = nn.Linear(n_input_features, 1)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred

model = LinearRegression(n_features)
learning_rate = 0.01
n_epoch = 100

# Loss and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(n_epoch):
    # forward pass and loss
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train) # y_pred has to come in first position and y_train in second
    
    # evaluation in progress
    accuracy = y_pred.round().eq(y_train).sum() / float(y_train.shape[0])

    # backward pass
    loss.backward()
    optimizer.step()

    # empty the grad history
    optimizer.zero_grad()

    if (epoch + 1) % 10 == 0:
        print(f" epoch : {epoch+1}  Loss : {loss.item():.5f}  Accuracy train data : {accuracy.item():.5f} ")

# Evaluation
with torch.no_grad():
    y_test_pred = model(X_test)
    y_test_pred_class = y_test_pred.round()
    accuracy = y_test_pred_class.eq(y_test).sum() / float(y_test.shape[0])
    print(f"Accuracy on test data : {accuracy.item()}")