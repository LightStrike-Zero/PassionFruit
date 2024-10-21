# train_model.py

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


# Load the YAML file
def load_yaml(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data

# Plotting function
def plot_and_save_metrics(train_losses, val_losses, train_accuracies, val_accuracies, num_epochs, l2_lambda):
    epochs = range(1, num_epochs + 1)

    fig, ax1 = plt.subplots()

    # Plot losses
    ax1.plot(epochs, train_losses, label="Training Loss", color="cornflowerblue")
    ax1.plot(epochs, val_losses, label="Validation Loss", color="orange", linestyle='--')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend(loc='upper right')

    # Plot accuracies on the same plot with a secondary y-axis
    # ax2 = ax1.twinx()
    # ax2.plot(epochs, train_accuracies, label="Training Accuracy", color="green")
    # ax2.plot(epochs, val_accuracies, label="Validation Accuracy", color="red", linestyle='--')
    # ax2.set_ylabel('Accuracy')
    # ax2.legend(loc='upper right')

    plt.title(f"Training and Validation Metrics\nL2 Regularisation: {l2_lambda}\nEpochs: {num_epochs}")

    # Add metadata: Time and date of the plot generation
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    plt.figtext(0.5, -0.1, f"Generated on: {now}", ha="center", fontsize=8)

    # Adjust spacing to prevent cutting off text
    plt.subplots_adjust(bottom=0.2)

    # Save the plot with the current date and time in the filename
    filename = f"metrics_curve_{now.replace(':', '-').replace(' ', '_')}.png"
    plt.savefig(filename, bbox_inches='tight')

    print(f"Metrics curve saved as {filename}")
    plt.show()


# Preprocess the data for training
# def preprocess_data(data, selected_features, target_stat='Result'):
#     rows = []
#     for match in data:
#         match_id = match['MatchId']
#         for team_key in ['Team1', 'Team2']:
#             team_data = match[team_key]
#             row = {'MatchId': match_id, 'Team': team_data['Name'], target_stat: 1 if team_data['Result'] == 'W' else 0}
#
#             # Only use selected features
#             row.update({key: value for key, value in team_data.items() if key in selected_features})
#             rows.append(row)
#
#     df = pd.DataFrame(rows)
#
#     # Convert to numeric where possible
#     numeric_columns = df.columns.difference(['Team', 'MatchId'])
#     df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
#
#     # Drop any rows with missing data
#     df.dropna(inplace=True)
#
#     return df

# Train the model
# Train the model
def train_model(data, model_save_path, scaler_save_path, l2_lambda, dropout_rate):
    # Preprocess data
    df = add_custom_features(data)

    # Separate features and target
    X = df[selected_features].values
    y = df['Result'].values

    # Standardize the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Save the scaler
    save_scaler(scaler, scaler_save_path)

    # Define K-Fold Cross-Validation
    k_folds = 5
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

    num_epochs_fold = num_epochs

    # To store metrics across all folds
    train_losses_folds = []
    val_losses_folds = []
    train_accuracies_folds = []
    val_accuracies_folds = []

    for fold, (train_index, val_index) in enumerate(skf.split(X, y)):
        print(f'Fold {fold + 1}/{k_folds}')
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # Convert to PyTorch tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

        # Initialize the model
        input_size = len(selected_features)
        model = WinPredictorMLP(input_size, dropout_rate=dropout_rate)

        # Define loss function and optimizer with L2 regularization (weight decay)
        criterion = torch.nn.BCELoss()
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=l2_lambda)

        # Use DataLoader for mini-batch gradient descent
        batch_size = 32
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # To store metrics per epoch
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

                # Collect predictions and targets
                preds = (outputs.detach() > 0.5).float()
                train_preds.extend(preds.cpu().numpy())
                train_targets.extend(batch_y.cpu().numpy())

            # Compute average training loss
            avg_train_loss = epoch_train_loss / len(train_loader)

            # Compute training accuracy
            train_accuracy = accuracy_score(train_targets, train_preds)

            # Validation step
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor)
                # Compute validation accuracy
                val_preds = (val_outputs > 0.5).float()
                val_accuracy = accuracy_score(y_val_tensor.cpu().numpy(), val_preds.cpu().numpy())

            # Store metrics
            train_losses.append(avg_train_loss)
            val_losses.append(val_loss.item())
            train_accuracies.append(train_accuracy)
            val_accuracies.append(val_accuracy)

            # Print progress every 10 epochs
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs_fold}], Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss.item():.4f}, Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}')

        # Store metrics per fold
        train_losses_folds.append(train_losses)
        val_losses_folds.append(val_losses)
        train_accuracies_folds.append(train_accuracies)
        val_accuracies_folds.append(val_accuracies)

        # Optionally, save the model for each fold
        # save_model(model, f"{model_save_path}_fold{fold+1}.pth")

    # Compute average metrics across folds
    avg_train_losses = np.mean(train_losses_folds, axis=0)
    avg_val_losses = np.mean(val_losses_folds, axis=0)
    avg_train_accuracies = np.mean(train_accuracies_folds, axis=0)
    avg_val_accuracies = np.mean(val_accuracies_folds, axis=0)

    # Plot the metrics
    plot_and_save_metrics(avg_train_losses, avg_val_losses, avg_train_accuracies, avg_val_accuracies, num_epochs_fold, l2_lambda)

    # Optionally, save the final model (e.g., the model from the last fold)
    save_model(model, model_save_path)


L2_lambda = 1e-4
num_epochs = 1000
Dropout_rate = 0.35
learning_rate = 0.001

# Main function
if __name__ == '__main__':
    yaml_file = '../match_stats_2024/clean_match_data_2024.yaml'
    model_save_path = 'trained_model.pth'
    scaler_save_path = 'scaler.pkl'

    # Load data from YAML file
    data = load_yaml(yaml_file)

    # Train the model
    train_model(data, model_save_path, scaler_save_path, L2_lambda, dropout_rate=Dropout_rate)
