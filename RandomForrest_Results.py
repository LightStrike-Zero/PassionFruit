import yaml
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Define file path
file_path = 'match_stats_2022/clean_match_data_2022.yaml'

# Load your YAML data file
try:
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
except FileNotFoundError:
    print(f"File not found: {file_path}")
    raise

# Parse the matches and team statistics
matches = []
for match in data:
    for team, stats in match.items():
        if team == 'MatchId':
            continue
        team_data = stats.copy()
        team_data['MatchId'] = match['MatchId']  # Add the match ID to the team's data
        matches.append(team_data)

# Create a DataFrame
df = pd.DataFrame(matches)

# Filter out rows where the result is a 'Draw' (D)
df = df[df['Result'] != 'D']

# Encode Result as binary (Win = 1, Loss = 0), ignoring any rows with 'D' (draw)
df['Result'] = df['Result'].map({'W': 1, 'L': 0})

df['Kicks_to_Handballs'] = df['Kicks'] / df['Handballs']
df['Efficiency_Weighted_Inside50s'] = df['DisposalEfficiency'] * df['Inside50s']
df['GoalAccuracy'] = df['Goals'] / (df['Goals'] + df['Behinds'])
df['ContestedPossessions_per_Disposal'] = df['ContestedPossessions'] / df['Disposals']
df['Goal_Accuracy_Inside50s'] = df['GoalAccuracy'] * df['Inside50s']
df['Goal_Accuracy_ShotsAtGoal'] = df['GoalAccuracy'] * df['ShotsAtGoal']

# Define features and target
features = ['Age', 'Experience', 'Kicks', 'Handballs', 'Disposals',
            'DisposalEfficiency', 'MetresGained', 'Inside50s', 'ContestedPossessions', 'GroundBallGets',
            'Intercepts', 'TotalClearances', 'Marks', 'ContestedMarks', 'InterceptMarks', 'ShotsAtGoal',
            'GoalAssists', 'Tackles', 'Hitouts',
            # Add combined features
            'Kicks_to_Handballs', 'Efficiency_Weighted_Inside50s',
            'GoalAccuracy', 'ContestedPossessions_per_Disposal',
            'Goal_Accuracy_Inside50s', 'Goal_Accuracy_ShotsAtGoal']

# Drop rows with missing 'Result' or feature values
df = df.dropna(subset=['Result'] + features)

# Prepare the feature matrix (X) and target vector (y)
X = df[features]
y = df['Result']

# Scale the features (standardization)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and test sets (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)

# Initialize the Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Perform 5-fold cross-validation on the training set
cv_scores = cross_val_score(model, X_train, y_train, cv=5)

# Print the cross-validation accuracies and mean accuracy
print(f'Cross-Validation Accuracies: {cv_scores}')
print(f'Mean Cross-Validation Accuracy: {cv_scores.mean()}')

# Train the Random Forest on the entire training set
model.fit(X_train, y_train)

# Evaluate the model on the test set
y_pred = model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Loss', 'Win']))

# Feature importance (how much each feature contributes to the prediction)
feature_importance = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_})
print("Feature Importance (sorted):")
print(feature_importance.sort_values(by='Importance', ascending=False))

# Summary of games played, wins, and losses (excluding draws)
total_games = len(df) // 2  # Total games (2 rows per match)
total_wins = df['Result'].sum()  # Summing the 'Result' column gives the number of wins
total_losses = len(df) - total_wins  # The rest are losses

print(f"Total games played (excluding draws): {total_games}")
print(f"Total wins: {total_wins}")
print(f"Total losses: {total_losses}")
