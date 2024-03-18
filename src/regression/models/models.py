from torch import nn
import torch
import logging
import timm
    
class GhostWindNet27(nn.Module):
    def __init__(self) -> None:        
        super(GhostWindNet27, self).__init__()
        self.embed = 70
        self.time_window = 27
        self.ghostnetv2 = timm.create_model('ghostnetv2_160', num_classes=self.embed, pretrained=False)
        self.head1 = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(self.time_window*(self.embed + 4), 70),
            nn.ReLU(),
            nn.BatchNorm1d(70),
            nn.Linear(70, 7),
        )

    def forward(self, X) -> torch.Tensor:
        X, pos = X
        b = X.shape[0]
        days = X.shape[2]
        X = torch.reshape(X, [b * days, X.shape[1], X.shape[3], X.shape[4]])
        X = self.ghostnetv2(X)
        pos = torch.reshape(pos, [b * days, 4])
        X = torch.cat((X, pos), 1)
        X = torch.reshape(X, [b, days * (self.embed + 4)])
        X = self.head1(X)
        return X


class GhostWindNet27(nn.Module):
    def __init__(self) -> None:        
        super(GhostWindNet27, self).__init__()
        self.embed = 70
        self.time_window = 27
        self.ghostnetv2 = timm.create_model('ghostnetv2_160', num_classes=self.embed, pretrained=False)
        self.head1 = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(self.time_window*(self.embed + 4), 70),
            nn.ReLU(),
            nn.BatchNorm1d(70),
            nn.Linear(70, 7),
        )

    def forward(self, X) -> torch.Tensor:
        X, pos = X
        b = X.shape[0]
        days = X.shape[2]
        X = torch.reshape(X, [b * days, X.shape[1], X.shape[3], X.shape[4]])
        X = self.ghostnetv2(X)
        pos = torch.reshape(pos, [b * days, 4])
        X = torch.cat((X, pos), 1)
        X = torch.reshape(X, [b, days * (self.embed + 4)])
        X = self.head1(X)
        return X
    


class GhostWindNet27(nn.Module):
    def __init__(self) -> None:        
        super(GhostWindNet27, self).__init__()
        self.embed = 70
        self.time_window = 27
        self.ghostnetv2 = timm.create_model('ghostnetv2_160', num_classes=self.embed, pretrained=False)
        self.head1 = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(self.time_window*(self.embed + 4), 70),
            nn.ReLU(),
            nn.BatchNorm1d(70),
            nn.Linear(70, 7),
        )

    def forward(self, X) -> torch.Tensor:
        X, pos = X
        b = X.shape[0]
        days = X.shape[2]
        X = torch.reshape(X, [b * days, X.shape[1], X.shape[3], X.shape[4]])
        X = self.ghostnetv2(X)
        pos = torch.reshape(pos, [b * days, 4])
        X = torch.cat((X, pos), 1)
        X = torch.reshape(X, [b, days * (self.embed + 4)])
        X = self.head1(X)
        return X


class BaselineQW(nn.Module):
    def __init__(self) -> None:        
        super(BaselineQW, self).__init__()
        self.dummy = nn.Linear(1, 1)
    def forward(self, X) -> torch.Tensor:
        X, pos = X
        
        X = X[:, 0, :, 1, 1]
        X = (X * 4.7078495) +  8.934635 + self.dummy(torch.tensor([1.1])) * 0.0
        Q = torch.stack([torch.quantile(X, q=0.96, interpolation='linear', axis=1),  
                         torch.quantile(X, q=0.85, interpolation='linear', axis=1),  
                         torch.quantile(X, q=0.70, interpolation='linear', axis=1),  
                         torch.quantile(X, q=0.50, interpolation='linear', axis=1),  
                         torch.quantile(X, q=0.25, interpolation='linear', axis=1),  
                         torch.quantile(X, q=0.15, interpolation='linear', axis=1),  
                         torch.quantile(X, q=0.05, interpolation='linear', axis=1)], axis=1)      
        
        return Q
    
class BaselineQT(nn.Module):
    def __init__(self) -> None:        
        super(BaselineQT, self).__init__()
        self.dummy = nn.Linear(1, 1)

    def forward(self, X) -> torch.Tensor:
        X, pos = X
        
        X = X[:, 1, :, 1, 1]
        X = (X * 20.841803) +  280.37646 - 271.15 + self.dummy(torch.tensor([1.1])) * 0.0
        Q = torch.stack([torch.quantile(X, q=0.96, interpolation='linear', axis=1),  
                          torch.quantile(X, q=0.85, interpolation='linear', axis=1),  
                          torch.quantile(X, q=0.70, interpolation='linear', axis=1),  
                          torch.quantile(X, q=0.50, interpolation='linear', axis=1),  
                          torch.quantile(X, q=0.25, interpolation='linear', axis=1),  
                          torch.quantile(X, q=0.15, interpolation='linear', axis=1),  
                          torch.quantile(X, q=0.05, interpolation='linear', axis=1)], axis=1)      
        return Q 
    

class BaselineLinear(nn.Module):
    def __init__(self) -> None:        
        super(BaselineLinear, self).__init__()
        self.Lin = nn.Sequential(
            nn.Linear(27, 27), 
            nn.BatchNorm1d(27),
            nn.Linear(27, 7),
        )

    def forward(self, X) -> torch.Tensor:
        X, pos = X
        
        X = X[:, 0, :, 1, 1]
        X = self.Lin(X)
        return X
    
