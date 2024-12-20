import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Define file path
file_path = '../match_stats_2022/clean_match_data_2022.yaml'

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

# Check for missing values in 'Result' and features
print("Missing values in 'Result':", df['Result'].isnull().sum())  # Check for NaNs in Result
print("Missing values in features:")
print(df.isnull().sum())  # Check for NaNs in all columns

df['Kicks_to_Handballs'] = df['Kicks'] / df['Handballs']
df['Efficiency_Weighted_Inside50s'] = df['DisposalEfficiency'] * df['Inside50s']
df['Goal_Accuracy'] = df['Goals'] / (df['Goals'] + df['Behinds'])
df['xScore_Differential'] = df['Score'] - df['xScore']
df['ContestedPossessions_per_Disposal'] = df['ContestedPossessions'] / df['Disposals']
df['Goal_Accuracy_Inside50s'] = df['Goal_Accuracy'] * df['Inside50s']
df['Goal_Accuracy_ShotsAtGoal'] = df['Goal_Accuracy'] * df['ShotsAtGoal']

# Define features and target
features = ['Kicks', 'Handballs', 'Disposals',
            'DisposalEfficiency', 'MetresGained', 'Inside50s', 'ContestedPossessions', 'GroundBallGets',
            'Intercepts', 'TotalClearances', 'Marks', 'ContestedMarks', 'InterceptMarks',
            'Tackles', 'Hitouts',
            # Add combined features
]

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

# Build the logistic regression model
model = LogisticRegression(max_iter=1000)  # Increasing iterations to ensure convergence
model.fit(X_train, y_train)

# Perform k-fold cross-validation (e.g., with 5 folds)
scores = cross_val_score(model, X_scaled, y, cv=5)

# Print the accuracy for each fold and the average accuracy
print(f'Cross-Validation Accuracies: {scores}')
print(f'Mean Accuracy: {scores.mean()}')



# Predict and evaluate the model on the test set
y_pred = model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Loss', 'Win']))

# Feature importance
feature_importance = pd.DataFrame({'Feature': features, 'Coefficient': model.coef_[0]})
print("Feature Importance (sorted):")
print(feature_importance.sort_values(by='Coefficient', ascending=False))

# Summary of games played, wins, and losses (ignoring draws)
total_games = len(df) // 2  # Total games (2 rows per match)
total_wins = df['Result'].sum()  # Summing the 'Result' column gives the number of wins
total_losses = len(df) - total_wins  # The rest are losses

print(f"Total games played (excluding draws): {total_games}")
print(f"Total wins: {total_wins}")
print(f"Total losses: {total_losses}")

