import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionModel(nn.Module):
    def __init__(self, num_patches, hidden_dim1=512, hidden_dim2=265):
        super(AttentionModel, self).__init__()
        self.fc1 = nn.Linear(num_patches, hidden_dim1)  
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)  
        self.fc3 = nn.Linear(hidden_dim2, 1)  

    def forward(self, attention):
        x = F.relu(self.fc1(attention))  #[batch_size, num_patches, num_patches]
        x = F.relu(self.fc2(x)) 
        output = self.fc3(x) 
        return output


# # Przykładowe dane
# batch_size = 20
# num_heads = 4
# num_patches = 5
# attention = torch.rand(batch_size, num_patches, num_patches)  # Przykładowy tensor atencji
# target = torch.rand(batch_size, 1)  # Przykładowy tensor target

# # Tworzenie modelu i obliczanie straty
# model = AttentionModel(num_patches)
# out = model(attention)
# print(out.shape)