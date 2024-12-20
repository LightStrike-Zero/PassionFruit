import yaml

# Define file path
file_path = '../match_stats_2022/clean_match_data_2022.yaml'

# Load your YAML data file
try:
    with open(file_path, 'r') as file:
        matches = yaml.safe_load(file)
except FileNotFoundError:
    print(f"File not found: {file_path}")
    raise

# Define an empty dictionary to hold the cumulative stats for each team
team_stats = {}
team_games_played = {}
team_last_two_games = {}

# Map fields from YAML data to output fields
stat_field_map = {
    'Score': 'Score',
    'Goals': 'Goals',
    'Behinds': 'Behinds',
    'Disposals': 'Disposals',
    'Inside50s': 'Inside50s',
    'Age': 'Age',  # We will treat this as an average, not cumulative
    'Experience': 'Experience',  # This too will be averaged
    'ScoreConceded': 'ScoreConceded',  # This is for the opponent's score
    'GoalsConceded': 'GoalsConceded',  # This is for the opponent's goals
    'BehindsConceded': 'BehindsConceded',  # This is for the opponent's behinds
}


# Function to update stats for a team
def update_team_stats(team, stats, opponent_stats):
    if team not in team_stats:
        # Initialize the team with a dictionary for cumulative stats
        team_stats[team] = {key: 0 for key in stat_field_map.values()}
        team_games_played[team] = 0  # Track games played by each team
        team_last_two_games[team] = []  # Track the last two games for each team

    # Increment games played by the team
    team_games_played[team] += 1

    # Update cumulative stats with match data (non-averaged stats)
    current_game_stats = {}
    for yaml_key, mapped_key in stat_field_map.items():
        if yaml_key in ['Age', 'Experience']:
            # Age and Experience are treated as averages, so we sum them temporarily for averaging later
            team_stats[team][mapped_key] += stats[yaml_key]
            current_game_stats[mapped_key] = stats[yaml_key]  # Save this game's data for the 2-game average
        elif yaml_key in stats and isinstance(stats[yaml_key], (int, float)):  # Only sum numerical stats
            team_stats[team][mapped_key] += stats[yaml_key]
            current_game_stats[mapped_key] = stats[yaml_key]  # Save this game's data for the 2-game average

    # Calculate "Score Conceded," "Goals Conceded," and "Behinds Conceded" from the opponent's stats
    team_stats[team]['ScoreConceded'] += opponent_stats['Score']
    team_stats[team]['GoalsConceded'] += opponent_stats['Goals']
    team_stats[team]['BehindsConceded'] += opponent_stats['Behinds']

    # Add the current game's stats to the last two games list
    current_game_stats['ScoreConceded'] = opponent_stats['Score']
    current_game_stats['GoalsConceded'] = opponent_stats['Goals']
    current_game_stats['BehindsConceded'] = opponent_stats['Behinds']

    if len(team_last_two_games[team]) >= 2:
        team_last_two_games[team].pop(0)  # Remove the oldest game if we already have 2
    team_last_two_games[team].append(current_game_stats)


# Function to calculate and output the desired fields
def output_team_stats(team):
    stats = team_stats[team]
    games_played = team_games_played[team]

    # Print team name
    print(f"{team} cumulative stats:")

    # Print total and average for each mapped field
    for key in stat_field_map.values():
        if key in ['Age', 'Experience']:
            # Age and Experience should be averaged
            avg = stats[key] / games_played if games_played > 0 else 0
            print(f"  {key}: Avg = {avg:.2f}")
        else:
            # For cumulative stats, print total and average
            total = stats[key]
            avg = total / games_played if games_played > 0 else 0
            print(f"  {key}: Total = {total}, Avg = {avg:.2f}")

    # Calculate and print the percentage based on total score and total score conceded
    if stats['ScoreConceded'] > 0:
        percentage = (stats['Score'] / stats['ScoreConceded']) * 100
    else:
        percentage = 0  # Avoid division by zero
    print(f"  Percentage: {percentage:.2f}")

    # Calculate and print the 2-game average
    print(f"\n{team} 2-game average stats:")
    for key in stat_field_map.values():
        total = 0
        count = len(team_last_two_games[team])
        if count > 0:
            for game in team_last_two_games[team]:
                total += game.get(key, 0)
            avg = total / count
            print(f"  {key}: Avg (last 2 games) = {avg:.2f}")
        else:
            print(f"  {key}: No data for last 2 games")

    # Calculate and print the 2-game percentage average
    total_score = sum(game.get('Score', 0) for game in team_last_two_games[team])
    total_score_conceded = sum(game.get('ScoreConceded', 0) for game in team_last_two_games[team])
    if total_score_conceded > 0:
        two_game_percentage = (total_score / total_score_conceded) * 100
    else:
        two_game_percentage = 0  # Avoid division by zero
    print(f"  Percentage (last 2 games): {two_game_percentage:.2f}\n")


# Iterate over each match in the loaded YAML data
for match in matches:
    team1 = match["Team1"]
    team2 = match["Team2"]

    # Update the stats for both teams, including the "Score Conceded"
    update_team_stats(team1["Name"], team1, team2)  # team1's opponent is team2
    update_team_stats(team2["Name"], team2, team1)  # team2's opponent is team1

# Output stats for each team
for team in team_stats:
    output_team_stats(team)
