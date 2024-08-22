import yaml
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Define file path
file_path = 'match_stats_2022/clean_match_data_2022.yaml'

# Function to load and prepare the data
def load_and_prepare_data(file_path):
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

    # Feature engineering: Create combined features
    df['Kicks_to_Handballs'] = df['Kicks'] / df['Handballs']
    df['Efficiency_Weighted_Inside50s'] = df['DisposalEfficiency'] * df['Inside50s']
    df['GoalAccuracy'] = df['Goals'] / (df['Goals'] + df['Behinds'])
    df['ContestedPossessions_per_Disposal'] = df['ContestedPossessions'] / df['Disposals']
    df['Goal_Accuracy_Inside50s'] = df['GoalAccuracy'] * df['Inside50s']
    df['Goal_Accuracy_ShotsAtGoal'] = df['GoalAccuracy'] * df['ShotsAtGoal']

    return df

# Function to create and train a Random Forest Regressor for any specified target stat
def random_forest_model(df, target_stat):
    # Define the features (all columns except the target stat and non-predictive columns)
    # Define features and target
    features = ['Kicks', 'Handballs', 'Disposals',
                'DisposalEfficiency', 'MetresGained', 'ContestedPossessions', 'GroundBallGets',
                'Intercepts', 'TotalClearances', 'Marks', 'ContestedMarks', 'InterceptMarks', 'Tackles',
                'Hitouts',
                'Kicks_to_Handballs', 'ContestedPossessions_per_Disposal']




    # Define the target variable (the specified stat)
    X = df[features]
    y = df[target_stat]

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Scale the features (standardization)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize the Random Forest Regressor
    rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)

    # Perform 5-fold cross-validation on the training set
    cv_scores = cross_val_score(rf_reg, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error')
    mean_cv_score = abs(cv_scores.mean())

    # Train the model on the entire training set
    rf_reg.fit(X_train_scaled, y_train)

    # Make predictions on the test data
    y_pred = rf_reg.predict(X_test_scaled)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Print evaluation metrics
    print(f"Random Forest Model for '{target_stat}'")
    print(f"Cross-Validation Mean Squared Error (CV MSE): {mean_cv_score}")
    print(f"Test Set Mean Squared Error: {mse}")
    print(f"R-Squared: {r2}")

    # Get the feature importance
    feature_importance = pd.DataFrame({'Feature': features, 'Importance': rf_reg.feature_importances_})

    # Sort and print feature importance
    print("Feature Importance (sorted):")
    print(feature_importance.sort_values(by='Importance', ascending=False))

    return rf_reg, feature_importance

# Load and prepare the data
file_path = 'match_stats_2022/clean_match_data_2022.yaml'
df = load_and_prepare_data(file_path)

# Example usage: Predicting 'ShotsAtGoal' with the Random Forest model
target_stat = 'GoalsFromTurnover'  # This can be any non-binary stat (e.g., 'Kicks', 'Disposals', etc.)
rf_model, importance = random_forest_model(df, target_stat)
