import os
import subprocess
from Utilities.GenerateTeamAverages import calculate_three_round_averages
from Utilities.EloRating import generate_elo_ratings  # Import the function directly
import requests
import json
import yaml
import sys


def download_json_files(year, rounds, folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    for round_number in range(1, rounds + 1):
        round_id = f"{year}{str(round_number).zfill(2)}"
        url = f"https://www.wheeloratings.com/src/match_stats/table_data/{round_id}.json" #api to fetch data

        try:
            response = requests.get(url)
            response.raise_for_status()
            file_path = os.path.join(folder_name, f"{year}_round_{str(round_number).zfill(2)}.json")
            with open(file_path, 'w') as file:
                file.write(response.text)
            print(f"Downloaded: {file_path}")
        except requests.exceptions.RequestException as e:
            print(f"Failed to download {url}: {e}")


def extract_team_data(folder_name):
    json_files = [f for f in os.listdir(folder_name) if f.endswith('.json')]

    all_matches = []
    for json_file in json_files:
        file_path = os.path.join(folder_name, json_file)
        with open(file_path, 'r') as file:
            data = json.load(file)
            team_data = data.get('TeamData', [])
            if not team_data:
                print(f"No TeamData in {json_file}, skipping.")
                continue
            # Extract relevant fields
            match_ids = team_data[0]['MatchId']
            teams = team_data[0]['Team']
            abbreviations = team_data[0]['Abbreviation']
            ages = team_data[0]['Age']
            experiences = team_data[0]['Experience']
            coaches_votes = team_data[0]['CoachesVotes']
            rating_points = team_data[0]['RatingPoints']
            kicks = team_data[0]['Kicks']
            handballs = team_data[0]['Handballs']
            disposals = team_data[0]['Disposals']
            disposal_efficiency = team_data[0]['DisposalEfficiency']
            metres_gained = team_data[0]['MetresGained']
            inside_50s = team_data[0]['Inside50s']
            contested_possessions = team_data[0]['ContestedPossessions']
            ground_ball_gets = team_data[0]['GroundBallGets']
            intercepts = team_data[0]['Intercepts']
            total_clearances = team_data[0]['TotalClearances']
            marks = team_data[0]['Marks']
            contested_marks = team_data[0]['ContestedMarks']
            intercept_marks = team_data[0]['InterceptMarks']
            shots_at_goal = team_data[0]['ShotsAtGoal']
            goals = team_data[0]['Goals']
            behinds = team_data[0]['Behinds']
            scores = team_data[0]['Score']
            goal_assists = team_data[0]['GoalAssists']
            tackles = team_data[0]['Tackles']
            hitouts = team_data[0]['Hitouts']
            matches = data.get('Matches', [])
            if not matches:
                print(f"No Matches data in {json_file}, skipping.")
                continue
            home_teams = matches[0]['HomeTeam']
            away_teams = matches[0]['AwayTeam']

            expected_matches = len(match_ids) // 2
            if len(home_teams) != expected_matches:
                print(f"Mismatch in number of matches in {json_file}, skipping.")
                continue

            for i in range(0, len(match_ids), 2):
                match_index = i // 2

                team1_name = teams[i]
                team2_name = teams[i + 1]

                home_team_name = home_teams[match_index]
                away_team_name = away_teams[match_index]

                team1_home = (team1_name == home_team_name)
                team2_home = (team2_name == home_team_name)

                team1_score = scores[i]
                team2_score = scores[i + 1]

                if team1_score > team2_score:
                    result_team1 = 'W'
                    result_team2 = 'L'
                elif team1_score < team2_score:
                    result_team1 = 'L'
                    result_team2 = 'W'
                else:
                    result_team1 = 'D'
                    result_team2 = 'D'

                match = {
                    'MatchId': match_ids[i],
                    'Team1': {
                        'Name': teams[i],
                        'Abbreviation': abbreviations[i],
                        'Home': team1_home,
                        'Result': result_team1,
                        'Age': ages[i],
                        'Experience': experiences[i],
                        'CoachesVotes': coaches_votes[i],
                        'RatingPoints': rating_points[i],
                        'Kicks': kicks[i],
                        'Handballs': handballs[i],
                        'Disposals': disposals[i],
                        'DisposalEfficiency': disposal_efficiency[i],
                        'MetresGained': metres_gained[i],
                        'Inside50s': inside_50s[i],
                        'ContestedPossessions': contested_possessions[i],
                        'GroundBallGets': ground_ball_gets[i],
                        'Intercepts': intercepts[i],
                        'TotalClearances': total_clearances[i],
                        'Marks': marks[i],
                        'ContestedMarks': contested_marks[i],
                        'InterceptMarks': intercept_marks[i],
                        'ShotsAtGoal': shots_at_goal[i],
                        'Goals': goals[i],
                        'Behinds': behinds[i],
                        'Score': scores[i],
                        'GoalAssists': goal_assists[i],
                        'Tackles': tackles[i],
                        'Hitouts': hitouts[i],
                    },
                    'Team2': {
                        'Name': teams[i + 1],
                        'Abbreviation': abbreviations[i + 1],
                        'Home': team2_home,
                        'Result': result_team2,
                        'Age': ages[i + 1],
                        'Experience': experiences[i + 1],
                        'CoachesVotes': coaches_votes[i + 1],
                        'RatingPoints': rating_points[i + 1],
                        'Kicks': kicks[i + 1],
                        'Handballs': handballs[i + 1],
                        'Disposals': disposals[i + 1],
                        'DisposalEfficiency': disposal_efficiency[i + 1],
                        'MetresGained': metres_gained[i + 1],
                        'Inside50s': inside_50s[i + 1],
                        'ContestedPossessions': contested_possessions[i + 1],
                        'GroundBallGets': ground_ball_gets[i + 1],
                        'Intercepts': intercepts[i + 1],
                        'TotalClearances': total_clearances[i + 1],
                        'Marks': marks[i + 1],
                        'ContestedMarks': contested_marks[i + 1],
                        'InterceptMarks': intercept_marks[i + 1],
                        'ShotsAtGoal': shots_at_goal[i + 1],
                        'Goals': goals[i + 1],
                        'Behinds': behinds[i + 1],
                        'Score': scores[i + 1],
                        'GoalAssists': goal_assists[i + 1],
                        'Tackles': tackles[i + 1],
                        'Hitouts': hitouts[i + 1]
                    }
                }
                all_matches.append(match)

    return all_matches


def save_to_yaml(matches, output_file):
    with open(output_file, 'w') as file:
        yaml.dump(matches, file, default_flow_style=False, sort_keys=False)

def acquire_data(year):
    rounds = 24
    folder_name = f'../match_stats_{year}'
    output_file = os.path.join(folder_name, f'clean_match_data_{year}.yaml')
    download_json_files(year, rounds, folder_name)
    matches = extract_team_data(folder_name)
    save_to_yaml(matches, output_file)
    print(f"Data saved to {output_file}")
    previous_year_folder = f'../match_stats_{int(year) - 1}'
    previous_year_file = os.path.join(previous_year_folder, f'clean_match_data_{int(year) - 1}.yaml')
    if not os.path.exists(previous_year_file):
        print(f"Previous year data not found. Downloading data for {int(year) - 1}.")
        download_json_files(int(year) - 1, rounds, previous_year_folder)
        previous_year_matches = extract_team_data(previous_year_folder)
        save_to_yaml(previous_year_matches, previous_year_file)
    calculate_three_round_averages(year)
    generate_elo_ratings(year)

if __name__ == "__main__":
    import sys
    year = sys.argv[1]
    acquire_data(year)
