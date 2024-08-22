import yaml

# Load the YAML file (replace 'data.yaml' with your actual file path)
file_path = 'match_stats_2022/clean_match_data_2022.yaml'

with open(file_path, 'r') as file:
    matches = yaml.safe_load(file)

# Function to determine which team has more contested possessions and display the result
def display_result_for_contested_possessions(matches):
    win_cp_and_win = 0
    lose_cp_and_win = 0
    match_count = 0

    for match in matches:
        match_count += 1
        team1 = match['Team1']
        team2 = match['Team2']

        team1_stat = team1['MetresGained'] + team1['ContestedPossessions'] + team1['ShotsAtGoal'] + team1['Inside50s']
        team2_stat = team2['MetresGained'] + team2['ContestedPossessions'] + team2['ShotsAtGoal'] + team2['Inside50s']

        # Calculate variance
        variance = abs(team1_stat - team2_stat)

        if team1_stat > team2_stat:
            print(f"Match {match['MatchId']}:\t\t{variance}\t{team1['Result']}")
            if team1['Result'] == 'W':
                win_cp_and_win += 1
            else:
                lose_cp_and_win += 1
        elif team2_stat > team1_stat:
            print(f"Match {match['MatchId']}:\t\t{variance}\t{team2['Result']}")
            if team2['Result'] == 'W':
                win_cp_and_win += 1
            else:
                lose_cp_and_win += 1
        else:
            print(f"Match {match['MatchId']}:\tBoth teams had the same Contested Possessions")

    # Display total wins and losses
    print("\nSummary:")
    print(f"Total Wins: {win_cp_and_win}")
    print(f"Total Losses: {lose_cp_and_win}")  # Assuming a win/loss outcome for each match

    # Calculate percentages
    win_cp_win_percentage = (win_cp_and_win / match_count) * 100
    lose_cp_win_percentage = (lose_cp_and_win / match_count) * 100

    # Display results
    print(f"Percentage of wins while winning Contested Possession: {win_cp_win_percentage:.2f}%")
    print(f"Percentage of wins while losing Contested Possession: {lose_cp_win_percentage:.2f}%")


# Display the results
display_result_for_contested_possessions(matches)
