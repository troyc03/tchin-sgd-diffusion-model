# Import model here.
from models.cnn import SimpleCNN
import torch

class ModelTrainer:
    def __init__(self, model, data_loader, loss_fn, optimizer):
        self.model = model
        self.data_loader = data_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer

    def train(self, num_epochs=10):
        for epoch in range(num_epochs):
            for batch in self.data_loader:
                inputs, targets = batch
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()   
    
    def evaluate(self, test_loader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in test_loader:
                inputs, targets = batch
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                total_loss += loss.item()
        average_loss = total_loss / len(test_loader)
        print(f'Average Test Loss: {average_loss}')
    
    
