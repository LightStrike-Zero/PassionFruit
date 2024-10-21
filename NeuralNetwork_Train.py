import yaml
import pickle
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Load the YAML file
def load_yaml(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data


# Preprocess the data into a DataFrame for training
def preprocess_data(data, target_stat='Result', selected_features=None):
    rows = []
    for match in data:
        match_id = match['MatchId']
        for team_key in ['Team1', 'Team2']:
            team_data = match[team_key]
            row = {'MatchId': match_id, 'Team': team_data['Name'], target_stat: 1 if team_data['Result'] == 'W' else 0}

            # Only keep selected features if provided
            if selected_features:
                row.update({key: value for key, value in team_data.items() if key in selected_features})
            else:
                row.update(
                    {key: value for key, value in team_data.items() if key not in ['Name', 'Abbreviation', 'Result']})

            rows.append(row)

    df = pd.DataFrame(rows)

    # Convert to numeric where possible
    numeric_columns = df.columns.difference(['Team', 'MatchId'])
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

    # Drop any rows with missing data
    df.dropna(inplace=True)

    return df


# Define the neural network model (same as before)
class WinPredictorMLP(nn.Module):
    def __init__(self, input_size):
        super(WinPredictorMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x


# Step 4: Train the model
def train_model(model, criterion, optimizer, train_loader, num_epochs=10000):
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss:.4f}')


# Step 5: Evaluate the model
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            predicted = (outputs >= 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f'Accuracy: {accuracy:.4f}')

# Save the trained model to disk
def save_model(model, path='trained_model.pth'):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


# Main function to run the training process
def main(yaml_file, model_save_path, scaler_save_path, selected_features=None):
    # Load data from YAML
    data = load_yaml(yaml_file)

    # Preprocess the data (same as before)
    df = preprocess_data(data, selected_features=selected_features)

    # Separate features and target
    X = df.drop(columns=['MatchId', 'Team', 'Result']).values
    y = df['Result'].values

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Save the scaler for later use
    with open(scaler_save_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to {scaler_save_path}")

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

    # Define dataset and data loader
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Initialize the model
    input_size = X_train.shape[1]
    model = WinPredictorMLP(input_size)

    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    print("\n--- Training the Model ---")
    train_model(model, criterion, optimizer, train_loader)

    # Save the trained model
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")




yaml_file = 'match_stats_2024/clean_match_data_2024.yaml'
model_save_path = 'trained_model.pth'
scaler_save_path = 'scaler.pkl'

# Define the features you want to use for training
selected_features = ['Kicks', 'Handballs', 'Disposals', 'DisposalEfficiency', 'MetresGained', 'Inside50s',
                     'ContestedPossessions', 'GroundBallGets', 'Intercepts', 'TotalClearances', 'Marks',
                     'ContestedMarks', 'InterceptMarks', 'Tackles', 'Hitouts']  # Example selected stats

# Run the training function with selected features
main(yaml_file, model_save_path, scaler_save_path, selected_features=selected_features)
