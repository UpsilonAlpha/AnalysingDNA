import numpy as np
import torch
import matplotlib.pyplot as plt


# Hyperparameters
learning_rate = 0.01
epochs = 10000

# These variables are used to control the specific example that we test
factor = 3
test_example = torch.tensor([10], dtype=torch.float32)


# Variables for our linear regression example
X = torch.tensor([[1],[2],[3],[4]], dtype=torch.float32)
Y = torch.tensor([[x * factor] for x in X], dtype=torch.float32)
w = torch.tensor(0.0, requires_grad=True)

num_samples, num_features = X.shape

input_size = num_features
output_size = num_features

class LinearRegression(nn.Module):
    
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()    
        self.lin = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        return self.lin(x)

model = LinearRegression(input_size, output_size)    


loss = nn.MSELoss()
optimiser = torch.optim.SGD(model.parameters(), lr=learning_rate)


print (f'Prediction before training = {test_example.item()} * {factor} = {model(test_example).item():.3f}\n')


# Training loop
for epoch in range(epochs):    
    
    # Forward pass
    y_pred = model(X)
    
    # Calculate Loss
    l = loss(y_pred, Y)
        
    # Calculate gradients
    l.backward()
    
    # Update weights
    optimiser.step()
    
    optimiser.zero_grad()
            
    if epoch % (epochs / 10) == 0:
        w = model.state_dict()['lin.weight']
        b = model.state_dict()['lin.bias']        
        print (f'Epoch {epoch+1}: weight = {w.item():.3f}, bias = {b.item():.3f}, loss = {l:.5f}')
        
with torch.inference_mode():
    prediction = model(test_example)
    print(f'Prediction after training = {test_example.item()} * {factor} = {prediction.item():.3f}')
    
    