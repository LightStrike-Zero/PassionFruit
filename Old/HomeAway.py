import yaml
from collections import OrderedDict

year = 2024  # Specify the year

# Custom YAML dumper to handle OrderedDict
def ordered_yaml_dump(data, stream=None, Dumper=yaml.Dumper, **kwargs):
    class OrderedDumper(Dumper):
        pass

    def _dict_representer(dumper, data):
        return dumper.represent_mapping(
            yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
            data.items()
        )

    OrderedDumper.add_representer(OrderedDict, _dict_representer)
    return yaml.dump(data, stream, OrderedDumper, **kwargs)

# Load match data from YAML file
with open(f'match_stats_{year}/clean_match_data_{year}.yaml', 'r') as file:
    matches = yaml.safe_load(file)

# Initialize counts
home_results = {'W': 0, 'L': 0, 'D': 0}

# Process each match
for match in matches:
    team1 = match['Team1']
    team2 = match['Team2']

    # Identify the home team
    if team1['Home']:
        home_team = team1
        away_team = team2
    elif team2['Home']:
        home_team = team2
        away_team = team1
    else:
        # If neither team is marked as home, skip this match
        print(f"Skipping match {match['MatchId']} as no home team is identified.")
        continue

    # Get the scores
    home_score = home_team['Score']
    away_score = away_team['Score']

    # Determine the result for the home team
    if home_score > away_score:
        result = 'W'  # Home team won
    elif home_score < away_score:
        result = 'L'  # Home team lost
    else:
        result = 'D'  # Draw

    # Increment the count
    home_results[result] += 1

# Calculate total matches
total_matches = sum(home_results.values())

# Calculate percentages and probabilities
percentages = {}
probabilities = {}
for result_type, count in home_results.items():
    percentage = (count / total_matches) * 100 if total_matches > 0 else 0
    probability = count / total_matches if total_matches > 0 else 0
    percentages[result_type] = round(percentage, 2)
    probabilities[result_type] = round(probability, 4)

# Use OrderedDict to maintain the order
output_data = OrderedDict([
    (f'HomeTeamResults{year}', OrderedDict([
        ('Wins', home_results['W']),
        ('Losses', home_results['L']),
        ('Draws', home_results['D']),
        ('TotalMatches', total_matches),
        ('Percentages', OrderedDict([
            ('Wins', percentages['W']),
            ('Losses', percentages['L']),
            ('Draws', percentages['D']),
        ])),
        ('Probabilities', OrderedDict([
            ('Wins', probabilities['W']),
            ('Losses', probabilities['L']),
            ('Draws', probabilities['D']),
        ])),
    ])),
])

# Output the results into a YAML file using the custom dumper
with open(f'home_team_results_{year}.yaml', 'w') as outfile:
    ordered_yaml_dump(output_data, outfile, default_flow_style=False)

print("Home team results have been saved.")

import yaml
from collections import OrderedDict

# Custom YAML dumper to handle OrderedDict
def ordered_yaml_dump(data, stream=None, Dumper=yaml.Dumper, **kwargs):
    class OrderedDumper(Dumper):
        pass

    def _dict_representer(dumper, data):
        return dumper.represent_mapping(
            yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
            data.items()
        )

    OrderedDumper.add_representer(OrderedDict, _dict_representer)
    return yaml.dump(data, stream, OrderedDumper, **kwargs)

# # Load match data from YAML file
# with open('match_stats_2023/clean_match_data_2023.yaml', 'r') as file:
#     matches = yaml.safe_load(file)

# Initialize team statistics
team_stats = OrderedDict()

for match in matches:
    team1 = match['Team1']
    team2 = match['Team2']

    # Get team names
    team1_name = team1['Name']
    team2_name = team2['Name']

    # Ensure teams are in team_stats
    if team1_name not in team_stats:
        team_stats[team1_name] = OrderedDict({
            'Home': {'Wins': 0, 'Losses': 0, 'Draws': 0, 'Total': 0, 'WinningPercentage': 0.0},
            'Away': {'Wins': 0, 'Losses': 0, 'Draws': 0, 'Total': 0, 'WinningPercentage': 0.0}
        })

    if team2_name not in team_stats:
        team_stats[team2_name] = OrderedDict({
            'Home': {'Wins': 0, 'Losses': 0, 'Draws': 0, 'Total': 0, 'WinningPercentage': 0.0},
            'Away': {'Wins': 0, 'Losses': 0, 'Draws': 0, 'Total': 0, 'WinningPercentage': 0.0}
        })

    # Identify home and away teams
    if team1['Home']:
        home_team = team1
        away_team = team2
        home_team_name = team1_name
        away_team_name = team2_name
    elif team2['Home']:
        home_team = team2
        away_team = team1
        home_team_name = team2_name
        away_team_name = team1_name
    else:
        # If neither team is marked as home, skip this match
        print(f"Skipping match {match['MatchId']} as no home team is identified.")
        continue

    # Get scores
    home_score = home_team['Score']
    away_score = away_team['Score']

    # Determine results
    if home_score > away_score:
        home_result = 'Wins'
        away_result = 'Losses'
    elif home_score < away_score:
        home_result = 'Losses'
        away_result = 'Wins'
    else:
        home_result = 'Draws'
        away_result = 'Draws'

    # Update home team stats
    team_stats[home_team_name]['Home'][home_result] += 1
    team_stats[home_team_name]['Home']['Total'] += 1

    # Update away team stats
    team_stats[away_team_name]['Away'][away_result] += 1
    team_stats[away_team_name]['Away']['Total'] += 1

# Calculate winning percentages
for team_name, stats in team_stats.items():
    # Home winning percentage
    home_total = stats['Home']['Total']
    if home_total > 0:
        home_wins = stats['Home']['Wins']
        stats['Home']['WinningPercentage'] = round((home_wins / home_total) * 100, 2)
    else:
        stats['Home']['WinningPercentage'] = 0.0

    # Away winning percentage
    away_total = stats['Away']['Total']
    if away_total > 0:
        away_wins = stats['Away']['Wins']
        stats['Away']['WinningPercentage'] = round((away_wins / away_total) * 100, 2)
    else:
        stats['Away']['WinningPercentage'] = 0.0

# Output the team statistics into a YAML file
with open('team_home_away_stats.yaml', 'w') as outfile:
    ordered_yaml_dump(team_stats, outfile, default_flow_style=False)

print("Team home and away statistics have been saved to 'team_home_away_stats.yaml'.")
