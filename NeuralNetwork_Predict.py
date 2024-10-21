import yaml
import pandas as pd
import torch
import torch.nn as nn
import pickle
from sklearn.preprocessing import StandardScaler

# Load the YAML file
def load_yaml(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data

# Preprocess the match data
def preprocess_match_data(data, selected_features):
    rows = []
    for match in data:
        match_id = match['MatchId']
        for team_key in ['Team1', 'Team2']:
            team_data = match[team_key]
            row = {'MatchId': match_id, 'Team': team_data['Name']}
            row.update({key: value for key, value in team_data.items() if key in selected_features})
            rows.append(row)

    df = pd.DataFrame(rows)
    numeric_columns = df.columns.difference(['Team', 'MatchId'])
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
    df.dropna(inplace=True)
    return df

# Step 3: Define the neural network model (needs to be in this script too)
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

# Load the trained model
def load_model(model_path, input_size):
    model = WinPredictorMLP(input_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print(f"Model loaded from {model_path}")
    return model

# Main function for prediction
def main(yaml_file, model_path, scaler_path, selected_features):
    data = load_yaml(yaml_file)
    df = preprocess_match_data(data, selected_features)

    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    print(f"Scaler loaded from {scaler_path}")

    X_matches = df[selected_features].values
    X_matches_scaled = scaler.transform(X_matches)

    X_matches_tensor = torch.tensor(X_matches_scaled, dtype=torch.float32)

    input_size = len(selected_features)
    model = load_model(model_path, input_size)

    for i, row in df.iterrows():
        match_id = row['MatchId']
        team_name = row['Team']

        with torch.no_grad():
            prediction = model(X_matches_tensor[i].unsqueeze(0))
            predicted_prob = prediction.item()
            predicted_winner = 1 if predicted_prob >= 0.5 else 0

            print(f"Match {match_id}: Team {predicted_winner + 1} ({team_name}) is predicted to win with probability {predicted_prob:.4f}")

# Prediction script paths
yaml_file = 'match_stats_2024/clean_match_data_2024.yaml'
model_path = 'trained_model.pth'
scaler_path = 'scaler.pkl'

selected_features = ['Kicks', 'Handballs', 'Disposals', 'DisposalEfficiency', 'MetresGained', 'Inside50s',
                     'ContestedPossessions', 'GroundBallGets', 'Intercepts', 'TotalClearances', 'Marks',
                     'ContestedMarks', 'InterceptMarks', 'Tackles', 'Hitouts']

main(yaml_file, model_path, scaler_path, selected_features)
