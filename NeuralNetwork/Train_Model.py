import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import TensorDataset, DataLoader
from MLP import WinPredictorMLP, selected_features, save_model, save_scaler, add_custom_features
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import yaml
import matplotlib.pyplot as plt
import datetime
import numpy as np


def load_yaml(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data

def plot_and_save_metrics(train_losses, val_losses, train_accuracies, val_accuracies, num_epochs, l2_lambda,
                          dropout_rate, learning_rate, start_year=2018, end_year=2023):
    epochs = range(1, num_epochs + 1)
    fig, ax1 = plt.subplots()
    ax1.plot(epochs, train_losses, label="Training Loss", color="cornflowerblue")
    ax1.plot(epochs, val_losses, label="Validation Loss", color="orange", linestyle='--')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend(loc='upper right')

    title_text = (
        "Training and Validation Metrics\n"
        f"Years Trained: {start_year}-{end_year}\n"
        f"L2 Regularisation: {l2_lambda}, Dropout Rate: {dropout_rate}\n"
        f"Learning Rate: {learning_rate}, Epochs: {num_epochs}"
    )
    plt.title(title_text, fontsize=10, pad=20)
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    plt.figtext(0.5, -0.1, f"Generated on: {now}", ha="center", fontsize=8)
    plt.subplots_adjust(top=0.8, bottom=0.2)
    filename = f"metrics_curve_{now.replace(':', '-').replace(' ', '_')}.png"
    plt.savefig(filename, bbox_inches='tight')
    print(f"Metrics curve saved as {filename}")
    plt.show()

# Train model
def train_model(data, model_save_path, scaler_save_path, l2_lambda, dropout_rate):
    df = add_custom_features(data)

    X = df[selected_features].values
    y = df['Result'].values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    save_scaler(scaler, scaler_save_path)

    # K-Fold
    k_folds = 5
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    num_epochs_fold = num_epochs
    train_losses_folds = []
    val_losses_folds = []
    train_accuracies_folds = []
    val_accuracies_folds = []

    for fold, (train_index, val_index) in enumerate(skf.split(X, y)):
        print(f'Fold {fold + 1}/{k_folds}')
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

        input_size = len(selected_features)
        model = WinPredictorMLP(input_size, dropout_rate=dropout_rate)

        criterion = torch.nn.BCELoss()
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=l2_lambda)

        batch_size = 32
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []

        for epoch in range(num_epochs_fold):
            model.train()
            epoch_train_loss = 0.0
            train_preds = []
            train_targets = []
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                train_loss = criterion(outputs, batch_y)
                train_loss.backward()
                optimizer.step()
                epoch_train_loss += train_loss.item()
                preds = (outputs.detach() > 0.5).float()
                train_preds.extend(preds.cpu().numpy())
                train_targets.extend(batch_y.cpu().numpy())

            avg_train_loss = epoch_train_loss / len(train_loader)
            train_accuracy = accuracy_score(train_targets, train_preds)

            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor)
                val_preds = (val_outputs > 0.5).float()
                val_accuracy = accuracy_score(y_val_tensor.cpu().numpy(), val_preds.cpu().numpy())

            train_losses.append(avg_train_loss)
            val_losses.append(val_loss.item())
            train_accuracies.append(train_accuracy)
            val_accuracies.append(val_accuracy)

            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs_fold}], Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss.item():.4f}, Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}')

        train_losses_folds.append(train_losses)
        val_losses_folds.append(val_losses)
        train_accuracies_folds.append(train_accuracies)
        val_accuracies_folds.append(val_accuracies)


    avg_train_losses = np.mean(train_losses_folds, axis=0)
    avg_val_losses = np.mean(val_losses_folds, axis=0)
    avg_train_accuracies = np.mean(train_accuracies_folds, axis=0)
    avg_val_accuracies = np.mean(val_accuracies_folds, axis=0)

    # Plot metrics
    plot_and_save_metrics(avg_train_losses, avg_val_losses, avg_train_accuracies, avg_val_accuracies, num_epochs_fold, l2_lambda, dropout_rate, learning_rate, start_year, end_year)

    # save the model
    save_model(model, model_save_path)


L2_lambda = 1e-3 #5e-4 #1e-5 #1e-4
num_epochs = 700 #500
dropout_rate = 0.35
learning_rate = 0.0003 #0.0005 #0.001
start_year = 2013
end_year = 2023

# Modify load_yaml to handle multiple files
def load_multiple_yaml_files(start_y, end_y):
    all_data = []
    for year in range(start_y, end_y + 1):
        file_path = f'../match_stats_{year}/clean_match_data_{year}.yaml'
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)
            all_data.extend(data)
    return all_data

# Main
if __name__ == '__main__':
    model_save_path = 'trained_model.pth'
    scaler_save_path = 'scaler.pkl'
    data = load_multiple_yaml_files(start_y=start_year, end_y=end_year)
    train_model(data, model_save_path, scaler_save_path, L2_lambda, dropout_rate=dropout_rate)
