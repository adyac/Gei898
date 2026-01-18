import torch.nn as nn

class Net(nn.Module):
    def __init__(self, K, D=9):
        super().__init__()
        H1 = 32
        H2 = 16
        H3 = H2
        H4 = H1
        self.encodeur = nn.Sequential(
            nn.Linear(D, H1),
            nn.ReLU(),
            nn.Linear(H1, H2),
            nn.ReLU(),
            nn.Linear(H2, K)
        )
        self.decodeur = nn.Sequential(
            nn.Linear(K, H3),
            nn.ReLU(),
            nn.Linear(H3, H4),
            nn.ReLU(),
            nn.Linear(H4, D)
        )
    def forward(self,x):
        latent_rep = self.encodeur(x)
        reconstruct = self.decodeur(latent_rep)
        return latent_rep,reconstruct




