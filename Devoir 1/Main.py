import numpy as np
import matplotlib.pyplot as plt
import torch
from Network import *
from Helpers import *
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

K = [7]
batch_size = 256
lr = 0.0015
epochs = 40
threshold = 0.02

train_data = np.loadtxt("./dataset/shuttle.trn", dtype=np.float32)
test_data = np.loadtxt("./dataset/shuttle.tst", dtype=np.float32)

# Split inputs / targets
train_inputs  = train_data[:, :-1]
train_targets = train_data[:, -1].astype(np.int64)

# Filter class 1
valid_train_idx    = np.where(train_targets == 1)[0]
valid_train_inputs = train_inputs[valid_train_idx]
valid_train_inputs, mu, sigma = normalize(valid_train_inputs)
test_inputs = (test_data[:, :-1] - mu) / sigma
valid_train_targets = train_targets[valid_train_idx]

# Convert to tensors
valid_train_inputs  = torch.from_numpy(valid_train_inputs)
valid_train_targets = torch.from_numpy(valid_train_targets).long()
test_inputs  = torch.from_numpy(test_inputs).float()
test_targets = torch.from_numpy(test_data[:, -1]).long()

print(f"Total train data given : {len(train_targets)}")
print(f"Valid train data : {len(valid_train_inputs)}")
print(f"thrown in trash : {len(train_targets) -len(valid_train_inputs) }")

valid_train_dataset = TensorDataset(valid_train_inputs, valid_train_targets)
valid_test_dataset  = TensorDataset(test_inputs, test_targets)

trainloader = torch.utils.data.DataLoader(valid_train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
testloader = torch.utils.data.DataLoader(valid_test_dataset, batch_size=1, shuffle=False, num_workers=0)

for k in K:
    train_loss_history = []
    test_loss_history = []
    model = Net(K=k).to(device)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for data, _ in trainloader:
            data = data.to(device)
            optimizer.zero_grad()

            data_latent, data_reconstruct = model(data)
            loss = loss_fn(data_reconstruct, data)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= len(trainloader)
        train_loss_history.append(epoch_loss)

        model.eval()
        valid_loss = 0
        print(f"#################################################\n"
              f"K = {k}\nepoch : {epoch + 1}     train loss :   {epoch_loss}\n")

    fig_loss_valid, ax_loss_valid = plt.subplots(1, 1)

    plot_epochs = np.arange(1, len(train_loss_history) + 1)

    ax_loss_valid.plot(plot_epochs, train_loss_history, label="Train loss")

    ax_loss_valid.set_xlabel("Epoch")
    ax_loss_valid.set_ylabel("MSE loss")
    ax_loss_valid.set_title(f"Loss curve (K={k})")
    ax_loss_valid.grid(True)
    plt.savefig(f'./figures/loss_curve_K{k}.png')
    plt.show()

    model.eval()
    test_loss = 0
    normal_losses = []
    abnormal_losses = []
    FP = 0
    VP = 0
    FN = 0
    VN = 0
    with torch.no_grad():
        for data, label in testloader:
            data = data.to(device)
            data_latent, data_reconstruct = model(data)
            loss = loss_fn(data_reconstruct, data)

            test_loss += loss.item()
            if loss.item() <= threshold:
                pred = "normal"
            else:
                pred = "abnormal" # abnormal

            if pred == "normal" and label.item() ==1 :
                VP += 1
            if pred == "normal" and label.item() != 1:
                FP += 1
            if pred == "abnormal" and label.item() == 1:
                FN += 1
            if pred == "abnormal" and label.item() != 1:
                VN +=1
            #for plotting
            if label.item() == 1:
                normal_losses.append(loss.item())
            else:
                abnormal_losses.append(loss.item())

        F1 = (2*VP)/(2*VP + FP + FN)
        accuracy = (VP+VN)/(VP+FP+FN+VN)
        print (f"###########################################################################\n"
               f"F score = {F1}\n"
               f"Accuracy = {accuracy*100}%")




        normal_losses = np.array(normal_losses)
        abnormal_losses = np.array(abnormal_losses)
        normal_losses = normal_losses[normal_losses <= 2.0]
        abnormal_losses = abnormal_losses[abnormal_losses <= 2.0]

        loss_max = max(normal_losses.max(), abnormal_losses.max())
        bins = np.linspace(0.0, 2.0, 1001)  # 100 intervals

        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].hist(normal_losses, bins=bins)
        ax[0].set_title("target = 1")
        ax[1].hist(abnormal_losses, bins=bins)
        ax[1].set_title("target != 1")
        for a in ax:
            a.set_xlabel("Reconstruction loss (MSE)")
            a.grid(True)
        ax[0].set_ylabel("Number of samples")
        plt.tight_layout()
        plt.show()


        test_loss /= len(testloader)
