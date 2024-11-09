import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

class Engine:
    def __init__(self, 
                 model: nn.Module,
                 train_data_loader: DataLoader,
                 test_data_loader: DataLoader,
                 loss_function: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 epochs: int,
                 accuracy):
        
        self.model = model
        self.train_data_loader = train_data_loader
        self.test_data_loader = test_data_loader
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.accuracy = accuracy
        self.epochs = epochs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    def train_step(self):
        self.model.to(self.device)
        self.model.train()
        train_loss, train_acc = 0, 0
        for batch, (X,y) in enumerate(self.train_data_loader):
            X,y = X.to(self.device), y.to(self.device)
            y_pred = self.model(X)
            loss = self.loss_function(y_pred,y)
            train_loss += loss.item()
            y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
            train_acc += self.accuracy(y_pred_class,y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        train_loss /= len(self.train_data_loader)
        train_acc /= len(self.train_data_loader)
        return train_loss, train_acc


    def test_step(self):
        self.model.to(self.device)
        self.model.eval()
        test_loss, test_acc = 0, 0
        with torch.inference_mode():
            for batch, (X,y) in enumerate(self.test_data_loader):
                X,y = X.to(self.device), y.to(self.device)
                test_pred = self.model(X)
                loss = self.loss_function(test_pred,y)
                test_loss += loss.item()
                test_pred_class = torch.argmax(torch.softmax(test_pred, dim=1), dim=1)
                test_acc += self.accuracy(test_pred_class,y)
            test_loss /= len(self.test_data_loader)
            test_acc /= len(self.test_data_loader)
        return test_loss, test_acc
    

    def training(self):
        
        results = {"train_loss": [],
                    "train_acc": [],
                    "test_loss": [],
                    "test_acc": []
                    }
        
        for epoch in tqdm(range(self.epochs)):
            train_loss, train_acc = self.train_step(model=self.model,
                                                    dataloader=self.train_data_loader,
                                                    loss_function=self.loss_function,
                                                    accuracy=self.accuracy,
                                                    optimizer=self.optimizer,
                                                    device=self.device)
            
            
            test_loss, test_acc = self.test_step(model=self.model,
                                                 dataloader=self.test_data_loader,
                                                 loss_function=self.loss_function,
                                                 accuracy=self.accuracy,
                                                 device=self.device)
            
            print(
                f"Epoch: {epoch+1} | "
                f"train_loss: {train_loss:.4f} | "
                f"train_acc: {train_acc:.4f} | "
                f"test_loss: {test_loss:.4f} | "
                f"test_acc: {test_acc:.4f}"
            )
            
            results["train_loss"].append(train_loss)
            results["train_acc"].append(train_acc)
            results["test_loss"].append(test_loss)
            results["test_acc"].append(test_acc)
            
        return results


        
