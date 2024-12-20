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
def preprocess_data(data, exclude_stats=None, target_stat='Inside50s'):
    rows = []
    for match in data:
        match_id = match['MatchId']
        for team_key in ['Team1', 'Team2']:
            team_data = match[team_key]
            row = {'MatchId': match_id, 'Team': team_data['Name'], target_stat: team_data[target_stat]}
            row.update({key: value for key, value in team_data.items() if key not in ['Name', 'Abbreviation', 'Result', target_stat]})
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


import pandas as pd

# Step 3: Perform Poisson Regression and output results to Excel and console
def perform_poisson_regression(df, target_stat, output_yaml, output_excel):
    # Define features (all columns except 'MatchId', 'Team', and target_stat)
    X = df.drop(columns=['MatchId', 'Team', target_stat])

    # Add constant (intercept) to the model
    X = sm.add_constant(X)

    # Target variable
    y = df[target_stat]

    # Fit Poisson regression model
    poisson_model = sm.GLM(y, X, family=sm.families.Poisson()).fit()

    # Extract the coefficient summary table from the model
    summary_table = poisson_model.summary().tables[1]

    # Convert summary_table to a Pandas DataFrame for sorting
    summary_df = pd.DataFrame(summary_table.data[1:], columns=summary_table.data[0])

    # Rename the first column (which may not be 'Unnamed: 0')
    first_column_name = summary_df.columns[0]
    summary_df = summary_df.rename(columns={first_column_name: 'Variable'})

    # Convert numeric columns to numbers, ignoring errors for non-numeric data
    summary_df = summary_df.apply(pd.to_numeric, errors='ignore')

    # Sort by 'coef' column
    sorted_summary = summary_df.sort_values(by='coef', ascending=False)

    # Prepare the result to save into YAML
    result_dict = sorted_summary.set_index('Variable').T.to_dict()

    # Write the result into YAML
    with open(output_yaml, 'w') as file:
        yaml.dump({f'{target_stat}_Poisson_Regression_Results': result_dict}, file)

    # Export sorted summary to Excel
    sorted_summary.to_excel(output_excel, index=False)

    # Print sorted results to the console
    print(f"\nPoisson Regression Coefficients (sorted by coefficient) for {target_stat}:")
    print(sorted_summary)

    # Print where results were saved
    print(f"\nPoisson Regression results saved to {output_yaml}")
    print(f"Poisson Regression results also saved to {output_excel}")

    return poisson_model

# Main function to run the analysis
def main(yaml_file, exclude_stats=None, target_stat='Inside50s', output_yaml='poisson_regression_results.yaml', output_excel='poisson_regression_results.xlsx'):
    # Load the YAML match data
    data = load_yaml(yaml_file)

    # Preprocess the data into a DataFrame
    df = preprocess_data(data, exclude_stats=exclude_stats, target_stat=target_stat)

    # Poisson Regression
    print(f"\n--- Poisson Regression for {target_stat} ---")
    poisson_model = perform_poisson_regression(df, target_stat, output_yaml, output_excel)



# Define the stats you want to exclude in a list
exclude_stats = ['ShotsAtGoal', 'Goals', 'Behinds', 'Score', 'xScore', 'xScoreRating', 'CoachesVotes', 'RatingPoints',
                 'GoalsFromKickIn', 'BehindsFromKickIn', 'PointsFromKickIn', 'GoalsFromStoppage', 'BehindsFromStoppage',
                 'PointsFromStoppage', 'GoalsFromTurnover', 'BehindsFromTurnover', 'Inside50s', 'GoalsFromDefensiveHalf',
                 'BehindsFromDefensiveHalf', 'PointsFromTurnover', 'GoalsFromForwardHalf', 'BehindsFromForwardHalf',
                 'PointsFromCentreBounce', 'GoalsFromCentreBounce', 'BehindsFromCentreBounce', 'PointsFromDefensiveHalf',
                 'GoalAssists']

year = '2024'
yaml_file = f'match_stats_{2024}/clean_match_data_{2024}.yaml'
target_stat = 'PointsFromForwardHalf'
output_yaml = f'regression/poisson_{target_stat}_{year}.yaml'
output_excel = f'regression/poisson_{target_stat}_{year}.xlsx'


# Run the main function
main(yaml_file, exclude_stats=exclude_stats, target_stat=target_stat, output_yaml=output_yaml, output_excel=output_excel)
