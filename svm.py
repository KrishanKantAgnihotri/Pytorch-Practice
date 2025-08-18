import torch 
import torch.nn as nn  # helps to develop neural network layers
import torch.optim as optim # contains optimization library likes SGD( used to train the model)

"""
Dummy Data preperation
"""

x = torch.tensor([[2.0,3.0],[1.0,1.0],[2.0,1.0],[3.0,2.0]]) # height and weight i.e input features
y = torch.tensor([1,-1,-1,1]) # +1 and -1 to seperate the classes
"""
Define SVM Model

nn.Module base class for all PyTorch models
nn.Linear(2,1) linear transformation y = wx+b 
forward Define how input flows through the model
"""
class SVM(nn.Module):
    def __init__(self):
        super(SVM,self).__init__()
        self.linear = nn.Linear(2,1) # input 2 features and output 1 feature

    def forward(self,x):
        return self.linear(x)
    
"""
Initialize Model , Loss and Optimizer 
model => our SVM model 
HingeEmbeddingLoss => loss function in SVM's
SGD: Stochastic Gradient Descent,a basic optimizer
"""

model = SVM()
criterion = nn.HingeEmbeddingLoss()
optimizer = optim.SGD(model.parameters(),lr=0.01)

"""
Training Loop

zero_grad() : clears old gradients
model(x) : Predicts ouptut
loss.backward() : Calculates gradients
optimizer.step() : Adjusts weights to reduce loss
"""

for epoch in range(100):
    optimizer.zero_grad()               # Clear previous gradients
    outputs= model(x).squeeze()         # forward pass
    loss = criterion(outputs,y.float()) # Compute loss
    loss.backward()                     # backpropagation
    optimizer.step()                    # Update weights


"""
Prediction
torch.no_grad()
tells Pytorch to not to track gradients(saves memory)
mdoel(test) : 
runs the model on new data
prediction.item() convert tensor to python number
"""

with torch.no_grad():
    test = torch.tensor([[2.5,2.0]])
    prediction = model(test)
    print("Prediction",prediction.item())
