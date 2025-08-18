import torch
import torch.nn as nn 
import torch.optim as optim
from torch.utils.data import DataLoader,Dataset
import torchvision.transforms as transforms

"""
Configuration Variables which can easily be configure when requried
"""
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 1e-3

"""
Custom Class to create Data set in pre defined format 
__init__ for data, labels, transform
__len__ to give length of data 
__getitem__ to creater like obj[idx]
"""
class CustomDataset(Dataset):
    def __init__(self,data,labels,transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform
    def __len__(self):
        return len(self.data)
    def __getitem__(self,idx):
        x =self.data[idx]
        y =self.labels[idx]
        if self.transform:
            x=self.transform(x)
        return x,y
"""
Custom defination for model
creating init with Linear , activation function , linear function
forward to pass the input to next layeers
"""
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel,self).__init__()
        self.net = nn.Sequential(
            nn.Linear(784,128),
            nn.ReLU(),
            nn.Linear(128,10)
        )
    def forward(self,x):
        return self.net(x)
    
"""
train function which is used in model training 
follow the step like 
- optimizer gradient reset
- get output for current input
- loss computation using loss function with current output and targets
- loss backward propogation to update the weights
- optimizer step to change the gradient for next inputs
"""
def train(model,dataloader,criterion,optimizer):
    model.train()
    for batch_idx,(inputs,targets) in enumerate(dataloader):
        inputs,targets = inputs.to(DEVICE),targets.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs,targets)
        loss.backward()
        optimizer.step()
"""
model.eval to evaluate the model with test data loader
torch.nograd so that gradient_descent information is not kept so less memory is utilized

"""
def evaluate(model,dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs,targets in dataloader:
            inputs,targets = inputs.to(DEVICE),targets.to(DEVICE)
            outputs = model(inputs)
            _,predicted = torch.max(outputs.data,1)
            total+=targets.size(0)
            correct+=(predicted==targets).sum().item()
    print(f"Accuracy: {100*correct/total:.2f}%")

def main():
    train_data = torch.randn(1000,784)
    train_labels = torch.randint(0,10,(1000,))
    test_data = torch.randn(200,784)
    test_labels = torch.randint(0,10,(200,))
    tranform = transforms.Compose([
        transforms.ToTensor()
    ])
    train_dataset = CustomDataset(train_data,train_labels,transform=None)
    test_dataset = CustomDataset(test_data,test_labels,tranform=None)
    train_loader = DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True)
    test_loader = DataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=True)
    model =MyModel().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=LEARNING_RATE)
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        train(model,train_loader,test_loader,criterion,optimizer)
        evaluate(model,test_loader)
if __name__=="__main__":
    main()
