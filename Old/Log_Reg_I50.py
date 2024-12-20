import yaml
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import statsmodels.api as sm


# Step 1: Load the YAML file
def load_yaml(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data


# Step 2: Preprocess the data into a Pandas DataFrame
def preprocess_data(data, exclude_stats=None):
    rows = []
    for match in data:
        match_id = match['MatchId']
        for team_key in ['Team1', 'Team2']:
            team_data = match[team_key]
            row = {'MatchId': match_id, 'Team': team_data['Name'], 'Inside50s': team_data['Inside50s']}
            row.update({key: value for key, value in team_data.items() if
                        key not in ['Name', 'Abbreviation', 'Result', 'Inside50s']})
            rows.append(row)

    df = pd.DataFrame(rows)

    # Exclude the stats specified in exclude_stats
    if exclude_stats:
        df.drop(columns=exclude_stats, inplace=True, errors='ignore')

    # Convert to numeric where possible, excluding non-numeric columns like 'Team'
    non_numeric_columns = ['Team']  # Exclude non-numeric columns
    numeric_columns = df.columns.difference(non_numeric_columns)  # Get numeric columns

    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')  # Convert numeric columns

    # Drop any rows with missing data
    df.dropna(inplace=True)

    return df


# Step 3: Perform Poisson Regression and sort coefficients by coefficient value
def perform_poisson_regression(df):
    # Define features (all columns except 'MatchId', 'Team', 'Inside50s')
    X = df.drop(columns=['MatchId', 'Team', 'Inside50s'])

    # Add constant (intercept) to the model
    X = sm.add_constant(X)

    # Target variable (Inside50s)
    y = df['Inside50s']

    # Fit Poisson regression model
    poisson_model = sm.GLM(y, X, family=sm.families.Poisson()).fit()

    # Extract the coefficient summary table from the model
    summary_table = poisson_model.summary().tables[1]

    # Convert summary_table to a Pandas DataFrame for sorting
    summary_df = pd.DataFrame(summary_table.data[1:], columns=summary_table.data[0])
    summary_df = summary_df.apply(pd.to_numeric, errors='ignore')

    # Sort by 'coef' column
    sorted_summary = summary_df.sort_values(by='coef', ascending=False)

    # Print the sorted summary
    print("\nPoisson Regression Coefficients (sorted by coefficient):")
    print(sorted_summary)

    return poisson_model


from sklearn.preprocessing import StandardScaler


def perform_logistic_regression(df):
    # Create a binary target variable for each match
    match_groups = df.groupby('MatchId')
    binary_rows = []

    for match_id, group in match_groups:
        team1 = group.iloc[0]
        team2 = group.iloc[1]
        if team1['Inside50s'] > team2['Inside50s']:
            binary_rows.append({'MatchId': match_id, 'Winner': team1['Team']})
        else:
            binary_rows.append({'MatchId': match_id, 'Winner': team2['Team']})

    binary_df = pd.DataFrame(binary_rows)

    # Merge predictions with original data
    df = df.merge(binary_df, on='MatchId')

    # Define features
    X = df.drop(columns=['MatchId', 'Team', 'Inside50s', 'Winner'])  # Features
    y = (df['Winner'] == df['Team']).astype(int)  # Binary target

    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Fit Logistic Regression model
    log_reg = LogisticRegression(max_iter=500)  # Increase max_iter to avoid convergence warning
    log_reg.fit(X_train, y_train)

    # Predict on test set
    y_pred = log_reg.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Logistic Regression Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    return log_reg


# Main function to run the analysis
def main(yaml_file, exclude_stats=None):
    data = load_yaml(yaml_file)
    df = preprocess_data(data, exclude_stats=exclude_stats)

    # Poisson Regression
    print("\n--- Poisson Regression ---")
    poisson_model = perform_poisson_regression(df)

    # Logistic Regression
    print("\n--- Logistic Regression ---")
    log_reg_model = perform_logistic_regression(df)


# Define the stats you want to exclude in a list
exclude_stats = ['ShotsAtGoal', 'Goals', 'Behinds', 'Score', 'xScore', 'xScoreRating', 'CoachesVotes', 'RatingPoints',
                 'GoalsFromKickIn',
                 'BehindsFromKickIn', 'PointsFromKickIn', 'GoalsFromStoppage', 'BehindsFromStoppage',
                 'PointsFromStoppage',
                 'GoalsFromTurnover', 'BehindsFromTurnover', 'PointsFromTurnover', 'GoalsFromDefensiveHalf',
                 'BehindsFromDefensiveHalf',
                 'PointsFromDefensiveHalf',
                 'GoalsFromForwardHalf',
                 'BehindsFromForwardHalf',
                 'PointsFromForwardHalf',
                 'GoalsFromCentreBounce',
                 'BehindsFromCentreBounce',
                 'PointsFromCentreBounce'
                 ]  # Example stats to exclude

# Replace with the actual path to your YAML file
yaml_file = '../match_stats_2024/clean_match_data_2024.yaml'
main(yaml_file, exclude_stats=exclude_stats)
