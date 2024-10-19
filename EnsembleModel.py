import yaml
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
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

# Initialize the base models
log_reg = LogisticRegression(max_iter=1000)
random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
xgboost = XGBClassifier(eval_metric='mlogloss')  # Removed "use_label_encoder"

# Create the Voting Classifier (ensemble)
ensemble = VotingClassifier(estimators=[
    ('log_reg', log_reg),
    ('random_forest', random_forest),
    ('xgboost', xgboost)
], voting='soft')  # 'hard' voting means majority voting, 'soft' uses predicted probabilities

# Perform 5-fold cross-validation on the training set
cv_scores = cross_val_score(ensemble, X_train, y_train, cv=5)

# Print the cross-validation accuracies and mean accuracy
print(f'Cross-Validation Accuracies: {cv_scores}')
print(f'Mean Cross-Validation Accuracy: {cv_scores.mean()}')

# Train the ensemble model on the entire training set
ensemble.fit(X_train, y_train)

# Evaluate the model on the test set
y_pred = ensemble.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Loss', 'Win']))

# Train Logistic Regression independently to get its coefficients
log_reg.fit(X_train, y_train)

# Coefficients from Logistic Regression (after fitting the model independently)
log_reg_coefs = pd.DataFrame({'Feature': features, 'Logistic_Regression_Coefficient': log_reg.coef_[0]})
print(log_reg_coefs.sort_values(by='Logistic_Regression_Coefficient', ascending=False))

# Train Random Forest separately if you need feature importance
random_forest.fit(X_train, y_train)

# Feature importance from Random Forest (after fitting the model independently)
rf_importances = pd.DataFrame({'Feature': features, 'Random_Forest_Importance': random_forest.feature_importances_})
print(rf_importances.sort_values(by='Random_Forest_Importance', ascending=False))

# Train XGBoost separately if you need feature importance
xgboost.fit(X_train, y_train)

# Feature importance from XGBoost (after fitting the model independently)
xgb_importances = pd.DataFrame({'Feature': features, 'XGBoost_Importance': xgboost.feature_importances_})
print(xgb_importances.sort_values(by='XGBoost_Importance', ascending=False))

# Summary of games played, wins, and losses (excluding draws)
total_games = len(df) // 2  # Total games (2 rows per match)
total_wins = df['Result'].sum()  # Summing the 'Result' column gives the number of wins
total_losses = len(df) - total_wins  # The rest are losses

print(f"Total games played (excluding draws): {total_games}")
print(f"Total wins: {total_wins}")
print(f"Total losses: {total_losses}")
