import sys

import yaml
import json
import os


def calculate_three_round_averages(target_year):
    previous_year = target_year - 1
    stats_file_path = f"../match_stats_{previous_year}/clean_match_data_{previous_year}.yaml"

    if not os.path.exists(stats_file_path):
        print(f"Stats file for {previous_year} not found at {stats_file_path}")
        return

    with open(stats_file_path, 'r') as file:
        stats_data = yaml.safe_load(file)

    team_stats = {}

    target_rounds = [24, 23, 22]

    for match in stats_data:
        match_id = match['MatchId']

        round_number = int(str(match_id)[4:6])

        if round_number in target_rounds:
            for team_key in ['Team1', 'Team2']:
                team_data = match[team_key]
                team_name = team_data['Name']

                if team_name not in team_stats:
                    team_stats[team_name] = {}

                for stat_key, stat_value in team_data.items():
                    if isinstance(stat_value, (int, float)):
                        if stat_key not in team_stats[team_name]:
                            team_stats[team_name][stat_key] = []
                        team_stats[team_name][stat_key].append(stat_value)

    team_averages = {}
    for team_name, stats in team_stats.items():
        team_averages[team_name] = {}
        for stat_key, values in stats.items():
            if values:
                team_averages[team_name][stat_key] = sum(values) / len(values)

    output_file = f"../match_stats_{previous_year}/average_stats_{previous_year}_last_3_rounds.json"
    with open(output_file, 'w') as json_file:
        json.dump(team_averages, json_file, indent=4)

    print(f"Averages for the last 3 rounds of {previous_year} saved to {output_file}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python GenerateTeamAverages.py <year>")
    else:
        year = int(sys.argv[1])
        calculate_three_round_averages(year)
