import os
import requests
import json
import yaml

def download_json_files(year, rounds, folder_name):
    # Create the folder if it doesn't exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    for round_number in range(1, rounds + 1):
        # Create the round_id
        round_id = f"{year}{str(round_number).zfill(2)}"

        # Construct the URL
        url = f"https://www.wheeloratings.com/src/match_stats/table_data/{round_id}.json"

        try:
            # Send the GET request to download the JSON file
            response = requests.get(url)
            response.raise_for_status()  # Raise an error for bad status codes

            # Save the JSON file in the folder with appropriate naming
            file_path = os.path.join(folder_name, f"{year}_round_{str(round_number).zfill(2)}.json")

            with open(file_path, 'w') as file:
                file.write(response.text)

            print(f"Successfully downloaded: {file_path}")

        except requests.exceptions.RequestException as e:
            print(f"Failed to download {url}: {e}")

# NOTE: It appears that devensive half scores had not been included prior to 2021 season

# Usage
year = 2024  # Specify the year
rounds = 24  # Number of rounds - use 28 if including the finals series (24 home and away rounds + 4 finals rounds)
folder_name = f'match_stats_{year}'  # Folder to save the JSON files

download_json_files(year, rounds, folder_name)




def extract_team_data(folder_name):
    # Get all the JSON files in the folder
    json_files = [f for f in os.listdir(folder_name) if f.endswith('.json')]

    all_matches = []

    for json_file in json_files:
        file_path = os.path.join(folder_name, json_file)

        # Open and load the JSON data
        with open(file_path, 'r') as file:
            data = json.load(file)

            # Extract TeamData
            team_data = data.get('TeamData', [])

            # Skip if TeamData is empty
            if not team_data:
                print(f"No TeamData in {json_file}, skipping.")
                continue

            # Extract relevant fields from TeamData
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
            # x_scores = team_data[0]['xScore']
            # x_score_rating = team_data[0]['xScoreRating']
            goal_assists = team_data[0]['GoalAssists']
            tackles = team_data[0]['Tackles']
            hitouts = team_data[0]['Hitouts']
            goals_from_kick_in = team_data[0]['GoalsFromKickIn']
            behinds_from_kick_in = team_data[0]['BehindsFromKickIn']
            points_from_kick_in = team_data[0]['PointsFromKickIn']
            goals_from_stoppage = team_data[0]['GoalsFromStoppage']
            behinds_from_stoppage = team_data[0]['BehindsFromStoppage']
            points_from_stoppage = team_data[0]['PointsFromStoppage']
            goals_from_turnover = team_data[0]['GoalsFromTurnover']
            behinds_from_turnover = team_data[0]['BehindsFromTurnover']
            points_from_turnover = team_data[0]['PointsFromTurnover']
            goals_from_defensive_half = team_data[0]['GoalsFromDefensiveHalf']
            behinds_from_defensive_half = team_data[0]['BehindsFromDefensiveHalf']
            points_from_defensive_half = team_data[0]['PointsFromDefensiveHalf']
            goals_from_forward_half = team_data[0]['GoalsFromForwardHalf']
            behinds_from_forward_half = team_data[0]['BehindsFromForwardHalf']
            points_from_forward_half = team_data[0]['PointsFromForwardHalf']
            goals_from_centre_bounce = team_data[0]['GoalsFromCentreBounce']
            behinds_from_centre_bounce = team_data[0]['BehindsFromCentreBounce']
            points_from_centre_bounce = team_data[0]['PointsFromCentreBounce']

            # Extract Matches data
            matches = data.get('Matches', [])
            if not matches:
                print(f"No Matches data in {json_file}, skipping.")
                continue

            # Extract HomeTeam and AwayTeam from Matches[0]
            home_teams = matches[0]['HomeTeam']
            away_teams = matches[0]['AwayTeam']

            # Check if lengths of lists match
            expected_matches = len(match_ids) // 2
            if len(home_teams) != expected_matches:
                print(f"Mismatch in number of matches in {json_file}, skipping.")
                continue

            # Pair the teams for each match
            for i in range(0, len(match_ids), 2):
                match_index = i // 2  # Calculate the match index

                team1_name = teams[i]
                team2_name = teams[i + 1]

                # Get the home and away team names for this match
                home_team_name = home_teams[match_index]
                away_team_name = away_teams[match_index]

                # Determine if team1 and team2 are home or away
                team1_home = (team1_name == home_team_name)
                team2_home = (team2_name == home_team_name)

                team1_score = scores[i]
                team2_score = scores[i + 1]

                if team1_score > team2_score:
                    result_team1 = 'W'  # Team 1 wins
                    result_team2 = 'L'  # Team 2 loses
                elif team1_score < team2_score:
                    result_team1 = 'L'  # Team 1 loses
                    result_team2 = 'W'  # Team 2 wins
                else:
                    result_team1 = 'D'  # Draw
                    result_team2 = 'D'  # Draw

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
                        # 'xScore': x_scores[i],
                        # 'xScoreRating': x_score_rating[i],
                        'GoalAssists': goal_assists[i],
                        'Tackles': tackles[i],
                        'Hitouts': hitouts[i],
                        'GoalsFromKickIn': goals_from_kick_in[i],
                        'BehindsFromKickIn': behinds_from_kick_in[i],
                        'PointsFromKickIn': points_from_kick_in[i],
                        'GoalsFromStoppage': goals_from_stoppage[i],
                        'BehindsFromStoppage': behinds_from_stoppage[i],
                        'PointsFromStoppage': points_from_stoppage[i],
                        'GoalsFromTurnover': goals_from_turnover[i],
                        'BehindsFromTurnover': behinds_from_turnover[i],
                        'PointsFromTurnover': points_from_turnover[i],
                        'GoalsFromDefensiveHalf': goals_from_defensive_half[i],
                        'BehindsFromDefensiveHalf': behinds_from_defensive_half[i],
                        'PointsFromDefensiveHalf': points_from_defensive_half[i],
                        'GoalsFromForwardHalf': goals_from_forward_half[i],
                        'BehindsFromForwardHalf': behinds_from_forward_half[i],
                        'PointsFromForwardHalf': points_from_forward_half[i],
                        'GoalsFromCentreBounce': goals_from_centre_bounce[i],
                        'BehindsFromCentreBounce': behinds_from_centre_bounce[i],
                        'PointsFromCentreBounce': points_from_centre_bounce[i]



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
                        # 'xScore': x_scores[i + 1],
                        # 'xScoreRating': x_score_rating[i + 1],
                        'GoalAssists': goal_assists[i + 1],
                        'Tackles': tackles[i + 1],
                        'Hitouts': hitouts[i + 1],
                        'GoalsFromKickIn': goals_from_kick_in[i + 1],
                        'BehindsFromKickIn': behinds_from_kick_in[i + 1],
                        'PointsFromKickIn': points_from_kick_in[i + 1],
                        'GoalsFromStoppage': goals_from_stoppage[i + 1],
                        'BehindsFromStoppage': behinds_from_stoppage[i + 1],
                        'PointsFromStoppage': points_from_stoppage[i + 1],
                        'GoalsFromTurnover': goals_from_turnover[i + 1],
                        'BehindsFromTurnover': behinds_from_turnover[i + 1],
                        'PointsFromTurnover': points_from_turnover[i + 1],
                        'GoalsFromDefensiveHalf': goals_from_defensive_half[i + 1],
                        'BehindsFromDefensiveHalf': behinds_from_defensive_half[i + 1],
                        'PointsFromDefensiveHalf': points_from_defensive_half[i + 1],
                        'GoalsFromForwardHalf': goals_from_forward_half[i + 1],
                        'BehindsFromForwardHalf': behinds_from_forward_half[i + 1],
                        'PointsFromForwardHalf': points_from_forward_half[i + 1],
                        'GoalsFromCentreBounce': goals_from_centre_bounce[i + 1],
                        'BehindsFromCentreBounce': behinds_from_centre_bounce[i + 1],
                        'PointsFromCentreBounce': points_from_centre_bounce[i + 1]
                    }
                }
                all_matches.append(match)

    return all_matches


def save_to_yaml(matches, output_file):
    # Write the match data to a YAML file
    with open(output_file, 'w') as file:
        yaml.dump(matches, file, default_flow_style=False, sort_keys=False)

# Usage
output_file = os.path.join(folder_name, f'clean_match_data_{year}.yaml')  # Output YAML file in the same folder

matches = extract_team_data(folder_name)
save_to_yaml(matches, output_file)

print(f"Match data successfully saved to {output_file}.")