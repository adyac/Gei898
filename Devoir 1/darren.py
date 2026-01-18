import numpy as np
from torch import nn

TRAIN_PATH = "dataset/shuttle.trn"
TEST_PATH = "dataset/shuttle.tst"

train_data = np.loadtxt(TRAIN_PATH)
test_data  = np.loadtxt(TEST_PATH)


X_train = train_data[:, 0:9]
y_train = train_data[:, 9].astype(int)
X_test = test_data[:, 0:9]
y_test = test_data[:, 9].astype(int)

X_train_sain = X_train[y_train == 1]
X_test_sain = X_test[y_test == 1]
X_test_fautif = X_test[y_test != 1]

print("Donnees train")
print(X_train)
print("\n" + "=" * 60)
print("Donnees train saines")
print(X_train_sain)
print("\n" + "=" * 60)
print(y_train)



print(f"Train:{len(X_train)}|  Train sain: {len(X_train_sain)} | Test sain: {len(X_test_sain)} | Test fautif: {len(X_test_fautif)}")

print("\n" + "=" * 60)
print("ETAPE 2: NORMALISATION")
print("=" * 60)

mu = X_train_sain.mean(axis=0)
sigma = X_train_sain.std(axis=0)
sigma[sigma == 0] = 1.0

X_train_norm = (X_train_sain - mu) / sigma
X_test_sain_norm = (X_test_sain - mu) / sigma
X_test_fautif_norm = (X_test_fautif - mu) / sigma

print(f"Moyenne apres normalisation: {X_train_norm.mean(axis=0).round(4)}")
print(f"Ecart-type apres normalisation: {X_train_norm.std(axis=0).round(4)}")


# AUTO-ENCODEUR
class AutoEncoder(nn.Module):
    def __init__(self, input_dim=9, latent_dim=4):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16), nn.ReLU(),
            nn.Linear(16, 8), nn.ReLU(),
            nn.Linear(8, latent_dim), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 8), nn.ReLU(),
            nn.Linear(8, 16), nn.ReLU(),
            nn.Linear(16, input_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

