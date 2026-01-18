
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

#train_path = "dataset/shuttle.trn"
#test_path = "dataset/shuttle.tst"

#train_data = np.loadtxt(TRAIN_PATH)
#test_data  = np.loadtxt(TEST_PATH)



# =============================================================================
# ÉTAPE 1: Chargement des données
# =============================================================================

def load_data(train_path, test_path):
    """Charge les données Shuttle depuis les fichiers .trn et .tst"""
    train_data = np.loadtxt(train_path)
    test_data = np.loadtxt(test_path)

    # Séparer features (9 capteurs) et labels (dernière colonne)
    X_train = train_data[:, :-1]
    y_train = train_data[:, -1].astype(int)

    X_test = test_data[:, :-1]
    y_test = test_data[:, -1].astype(int)

    return X_train, y_train, X_test, y_test


def explore_data(X_train, y_train, X_test, y_test):
    """Affiche les statistiques des données"""
    print("=" * 60)
    print("EXPLORATION DES DONNÉES")
    print("=" * 60)

    print(f"\nEntraînement: {X_train.shape[0]} échantillons, {X_train.shape[1]} capteurs")
    print(f"Test: {X_test.shape[0]} échantillons")

    print(f"\n--- Distribution des classes (Entraînement) ---")
    for classe in sorted(np.unique(y_train)):
        count = np.sum(y_train == classe)
        status = "SAIN" if classe == 1 else "FAUTIF"
        print(f"  Classe {classe} ({status}): {count} ({100 * count / len(y_train):.2f}%)")


# =============================================================================
# ÉTAPE 2: Normalisation des données
# =============================================================================

def normalize_data(X_train_sain, X_test):
    """
    Normalise les données avec moyenne et écart-type des données saines d'entraînement.
    x' = (x - mu) / sigma
    """
    # Calculer moyenne et std sur les données SAINES d'entraînement uniquement
    mu = X_train_sain.mean(axis=0)
    sigma = X_train_sain.std(axis=0)

    # Éviter division par zéro
    sigma[sigma == 0] = 1.0

    # Normaliser
    X_train_norm = (X_train_sain - mu) / sigma
    X_test_norm = (X_test - mu) / sigma

    return X_train_norm, X_test_norm, mu, sigma


# =============================================================================
# ÉTAPE 3: Définition de l'Auto-encodeur
# =============================================================================

class AutoEncoder(nn.Module):
    """Auto-encodeur pour détection d'anomalies"""

    def __init__(self, input_dim=9, latent_dim=4):
        super(AutoEncoder, self).__init__()

        # Encodeur: 9 -> 16 -> 8 -> latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, latent_dim),
            nn.ReLU()
        )

        # Décodeur: latent_dim -> 8 -> 16 -> 9
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def encode(self, x):
        return self.encoder(x)


# =============================================================================
# ÉTAPE 4: Entraînement
# =============================================================================

def train_autoencoder(model, train_loader, epochs=100, lr=0.001, device='cpu'):
    """Entraîne l'auto-encodeur sur les données saines"""

    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    losses = []

    print("\n" + "=" * 60)
    print("ENTRAÎNEMENT DE L'AUTO-ENCODEUR")
    print("=" * 60)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for batch_x, in train_loader:
            batch_x = batch_x.to(device)

            # Forward
            x_hat = model(batch_x)
            loss = criterion(x_hat, batch_x)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}] - Loss: {avg_loss:.6f}")

    return losses


# =============================================================================
# ÉTAPE 5: Calcul des erreurs de reconstruction
# =============================================================================

def compute_reconstruction_errors(model, X, device='cpu'):
    """Calcule l'erreur de reconstruction (MSE) pour chaque échantillon"""
    model.eval()

    with torch.no_grad():
        X_tensor = torch.FloatTensor(X).to(device)
        X_hat = model(X_tensor)

        # Erreur MSE par échantillon
        errors = ((X_tensor - X_hat) ** 2).mean(dim=1).cpu().numpy()

    return errors


# =============================================================================
# ÉTAPE 6: Calcul de la F-mesure et recherche du seuil optimal
# =============================================================================

def compute_f1_score(y_true, y_pred):
    """
    Calcule la F1-mesure.
    Positif = anomalie (fautif), Négatif = normal (sain)
    """
    VP = np.sum((y_pred == 1) & (y_true == 1))  # Vrais Positifs
    FP = np.sum((y_pred == 1) & (y_true == 0))  # Faux Positifs
    FN = np.sum((y_pred == 0) & (y_true == 1))  # Faux Négatifs

    if (2 * VP + FP + FN) == 0:
        return 0.0

    f1 = (2 * VP) / (2 * VP + FP + FN)
    return f1


def find_optimal_threshold(errors_sain, errors_fautif, num_thresholds=1000):
    """
    Trouve le seuil optimal qui maximise la F1-mesure.
    """
    # Créer labels: 0 = sain, 1 = fautif (anomalie)
    y_true = np.concatenate([
        np.zeros(len(errors_sain)),
        np.ones(len(errors_fautif))
    ])

    all_errors = np.concatenate([errors_sain, errors_fautif])

    # Tester différents seuils
    thresholds = np.linspace(all_errors.min(), all_errors.max(), num_thresholds)
    f1_scores = []

    for threshold in thresholds:
        # Prédiction: erreur >= seuil -> anomalie (1)
        y_pred = (all_errors >= threshold).astype(int)
        f1 = compute_f1_score(y_true, y_pred)
        f1_scores.append(f1)

    f1_scores = np.array(f1_scores)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]

    return best_threshold, best_f1, thresholds, f1_scores


# =============================================================================
# ÉTAPE 7: Visualisation
# =============================================================================

def plot_results(losses, thresholds, f1_scores, errors_sain, errors_fautif,
                 best_threshold, best_f1):
    """Génère les graphiques pour la présentation"""

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Courbe de loss
    axes[0, 0].plot(losses)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss (MSE)')
    axes[0, 0].set_title('Courbe d\'entraînement')
    axes[0, 0].grid(True)

    # 2. Distribution des erreurs
    axes[0, 1].hist(errors_sain, bins=50, alpha=0.7, label='Sain', density=True)
    axes[0, 1].hist(errors_fautif, bins=50, alpha=0.7, label='Fautif', density=True)
    axes[0, 1].axvline(x=best_threshold, color='r', linestyle='--',
                       label=f'Seuil optimal: {best_threshold:.4f}')
    axes[0, 1].set_xlabel('Erreur de reconstruction')
    axes[0, 1].set_ylabel('Densité')
    axes[0, 1].set_title('Distribution des erreurs de reconstruction')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # 3. F-mesure en fonction du seuil
    axes[1, 0].plot(thresholds, f1_scores)
    axes[1, 0].axvline(x=best_threshold, color='r', linestyle='--')
    axes[1, 0].axhline(y=best_f1, color='g', linestyle='--',
                       label=f'F1 max: {best_f1:.4f}')
    axes[1, 0].set_xlabel('Seuil T₀')
    axes[1, 0].set_ylabel('F1-mesure')
    axes[1, 0].set_title('F-mesure en fonction du seuil')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # 4. Texte récapitulatif
    axes[1, 1].axis('off')
    summary = f"""
    RÉSULTATS
    =========

    Seuil optimal T₀: {best_threshold:.6f}

    F1-mesure: {best_f1:.4f} ({best_f1 * 100:.2f}%)

    Objectif atteint: {'OUI ✓' if best_f1 > 0.90 else 'NON ✗'}
    (objectif: F1 > 90%)
    """
    axes[1, 1].text(0.1, 0.5, summary, fontsize=14, family='monospace',
                    verticalalignment='center')

    plt.tight_layout()
    plt.savefig('resultats_autoencoder.png', dpi=150)
    plt.show()

    print(f"\nGraphiques sauvegardés dans 'resultats_autoencoder.png'")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Configuration
    TRAIN_PATH = "dataset/shuttle.trn"
    TEST_PATH = "dataset/shuttle.tst"
    LATENT_DIM = 4
    EPOCHS = 10
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001

    device = torch.device('mps' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # -------------------------------------------------------------------------
    # ÉTAPE 1: Chargement des données
    # -------------------------------------------------------------------------
    X_train, y_train, X_test, y_test = load_data(TRAIN_PATH, TEST_PATH)
    explore_data(X_train, y_train, X_test, y_test)

    # Séparer données saines et fautives
    X_train_sain = X_train[y_train == 1]
    X_test_sain = X_test[y_test == 1]
    X_test_fautif = X_test[y_test != 1]

    print(f"\nDonnées saines pour entraînement: {len(X_train_sain)}")
    print(f"Données saines pour test: {len(X_test_sain)}")
    print(f"Données fautives pour test: {len(X_test_fautif)}")

    # -------------------------------------------------------------------------
    # ÉTAPE 2: Normalisation
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("NORMALISATION DES DONNÉES")
    print("=" * 60)

    X_train_norm, _, mu, sigma = normalize_data(X_train_sain, X_train_sain)
    X_test_sain_norm = (X_test_sain - mu) / sigma
    X_test_fautif_norm = (X_test_fautif - mu) / sigma

    print(f"Moyenne (mu): {mu}")
    print(f"Écart-type (sigma): {sigma}")

    # -------------------------------------------------------------------------
    # ÉTAPE 3: Préparer DataLoader
    # -------------------------------------------------------------------------
    train_tensor = torch.FloatTensor(X_train_norm)
    train_dataset = TensorDataset(train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # -------------------------------------------------------------------------
    # ÉTAPE 4: Créer et entraîner le modèle
    # -------------------------------------------------------------------------
    model = AutoEncoder(input_dim=9, latent_dim=LATENT_DIM)
    print(f"\nArchitecture du modèle:\n{model}")

    losses = train_autoencoder(model, train_loader, epochs=EPOCHS,
                               lr=LEARNING_RATE, device=device)

    # -------------------------------------------------------------------------
    # ÉTAPE 5: Calculer les erreurs de reconstruction
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("CALCUL DES ERREURS DE RECONSTRUCTION")
    print("=" * 60)

    errors_sain = compute_reconstruction_errors(model, X_test_sain_norm, device)
    errors_fautif = compute_reconstruction_errors(model, X_test_fautif_norm, device)

    print(f"Erreurs saines - Moyenne: {errors_sain.mean():.6f}, Std: {errors_sain.std():.6f}")
    print(f"Erreurs fautives - Moyenne: {errors_fautif.mean():.6f}, Std: {errors_fautif.std():.6f}")

    # -------------------------------------------------------------------------
    # ÉTAPE 6: Trouver le seuil optimal
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("RECHERCHE DU SEUIL OPTIMAL")
    print("=" * 60)

    best_threshold, best_f1, thresholds, f1_scores = find_optimal_threshold(
        errors_sain, errors_fautif
    )

    print(f"\nSeuil optimal T₀: {best_threshold:.6f}")
    print(f"F1-mesure maximale: {best_f1:.4f} ({best_f1 * 100:.2f}%)")
    print(f"\nObjectif (F1 > 90%): {'ATTEINT ✓' if best_f1 > 0.90 else 'NON ATTEINT ✗'}")

    # -------------------------------------------------------------------------
    # ÉTAPE 7: Visualisation
    # -------------------------------------------------------------------------
    plot_results(losses, thresholds, f1_scores, errors_sain, errors_fautif,
                 best_threshold, best_f1)