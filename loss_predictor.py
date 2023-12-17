import torch
import torch.nn as nn

class LinearBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()

        self.block = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor):
        return self.block(x)

class LossPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        # self.fc1 = nn.Linear(num_patches, hidden_dim1)  
        # self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)  
        # self.fc3 = nn.Linear(hidden_dim2, 1)

        self.model = nn.Sequential(
            LinearBlock(1, 256),
            LinearBlock(256, 512),
            LinearBlock(512, 256),
            LinearBlock(256, 1),
        )

    def forward(self, attention: torch.Tensor):
        # [batch_size, query_len, key_len]
        x = attention.unsqueeze(-1)

        x = self.model(x)

        aggregated_x = torch.mean(x, dim=2).squeeze(-1)

        return aggregated_x