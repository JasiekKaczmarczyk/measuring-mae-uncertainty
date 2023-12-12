import torch.nn as nn
import torch.nn.functional as F

class AttentionModel(nn.Module):
    def __init__(self, num_heads, hidden_dim1, hidden_dim2):
        super(AttentionModel, self).__init__()
        self.fc1 = nn.Linear(num_heads, hidden_dim1)  
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)  
        self.fc3 = nn.Linear(hidden_dim2, 1)  

    def forward(self, attention):
        attention_mean = attention.mean(dim=-1).mean(dim=-1)
        x = F.relu(self.fc1(attention_mean))  
        x = F.relu(self.fc2(x)) 
        output = self.fc3(x) 
        return output

def calculate_loss(model, attention, target):
    predictions = model(attention)
    loss = F.mse_loss(predictions, target, reduction='none')
    return loss.mean(dim=1) 