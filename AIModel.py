import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import plotly.express as px


#Reading data
dnameth = pd.read_csv("G9_liver_dna-meth.csv")
rnaseq = pd.read_csv("G9_liver_gene-expr.csv")

#Pre-processing data
dnameth.dropna()
rnaseq.dropna()
dnameth.iloc[:, 1].unique()
dnameth = dnameth.rename(columns = {'Primary Tumor':'0'})
rnaseq.iloc[:, 1].unique()
rnaseq.rename(columns = {'Primary Tumor':'0'})

#Defining dependant variable
y = dnameth['Label'].values

#Encoding categorical data
Labelencoder = LabelEncoder()
Y_encoding = Labelencoder.fit_transform(y) # Primary Tumor = 0, Solid Tissue Normal = 1

# Define the independent variables to drop the Label and Unnamed: 0
X = dnameth.drop(labels = ['Label','Unnamed: 0'], axis = 1)
feature_names = np.array(X.columns)

#Normalisation
scaler = StandardScaler()
scaler.fit(X)
X_scale = scaler.transform(X)
X_log2 = np.log2(X)
X_log2 = scaler.fit_transform(X_log2)


X_train, X_test, Y_train, Y_test = train_test_split(X_scale, Y_encoding, test_size = 0.3, random_state = 42)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

loss_over_time = []


# Hyperparameters
learning_rate = 0.00015
epochs = 1000

# Variables for controlling the test example
test_example = torch.tensor(X_test, dtype=torch.float32)


# Variables
X = torch.tensor(X_train, dtype=torch.float32)
Y = torch.tensor(Y_train, dtype=torch.float32)
w = torch.tensor(0.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)


num_samples, num_features = X.shape


input_size = num_features
output_size = num_samples

class Detector(nn.Module):
    
    def __init__(self, input_dim, output_dim):
        super(Detector, self).__init__()    
        self.lin = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        return self.lin(x)

model = Detector(input_size, output_size)    


loss = nn.MSELoss()
optimiser = torch.optim.SGD(model.parameters(), lr=learning_rate)


#print (f'Prediction before training = {test_example.item()} * {factor} = {model(test_example).item():.3f}\n')


# Training loop
for epoch in range(epochs):

    total_loss = 0
    
    for i in range(num_samples):

        # Forward pass
        y_pred = model(X[i])
    
        # Calculate Loss
        l = loss(y_pred, Y[i])
        
        # Calculate gradients
        l.backward()

        total_loss += float(l)
    
    print(total_loss)
    loss_over_time.append(total_loss)
    # Update weights
    optimiser.step()
    optimiser.zero_grad()
            
    if epoch % (epochs / 10) == 0:
        w = model.state_dict()['lin.weight']
        b = model.state_dict()['lin.bias']        
        #print (f'Epoch {epoch+1}: weight = {w.item():.3f}, bias = {b.item():.3f}, loss = {l:.5f}')
        
with torch.inference_mode():
    prediction = model(test_example)
    print(prediction)
    #print(f'Prediction after training = {test_example.item()} * {factor} = {prediction.item():.3f}')
    
graph = px.line(loss_over_time)
graph.show()