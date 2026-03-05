import torch.nn as nn
import torch.nn.functional as F

class Encoder_model(nn.Module):
    def __init__(self):
        super(Encoder_model,self).__init__()
        self.net = nn.Sequential(
            nn.Linear(28*28, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
            nn.ReLU()
        )
        self.has_printed_shape = False


    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten
        x = self.net(x)
        if not self.has_printed_shape:
            print(f"Encoder model latent shape: {x.shape}")
            self.has_printed_shape=True

        return x  # shared latent representation

# Client-specific task head (classification)
class TaskHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TaskHead, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 4), # Reducing sie of hidden layer from 50 to 4
            nn.ReLU(),
            nn.Linear(4, output_dim)   # Reducing sie of hidden layer from 50 to 4
        )

    def forward(self, x):
        return self.fc(x)

# Combined model: encoder structure is same for all clients, head is client-specific
class ClientModel(nn.Module):
    def __init__(self, encoder, output_dim):
        super(ClientModel, self).__init__()
        self.encoder = encoder
        self.head = TaskHead(input_dim=8, output_dim=output_dim)  # 8 is output size from encoder

    def forward(self, x):
        z = self.encoder(x)
        logits = self.head(z)
        return logits
