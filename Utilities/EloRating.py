import yaml
from collections import defaultdict
import sys

def expected_score(rating_a, rating_b):
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

def update_rating(rating, expected, actual, k):
    return rating + k * (actual - expected)

def normalize_ratings(ratings, min_rating, max_rating):
    original_min = min(ratings.values())
    original_max = max(ratings.values())
    normalized_ratings = {}
    for team, rating in ratings.items():
        normalized_value = min_rating + ((rating - original_min) * (max_rating - min_rating)) / (original_max - original_min)
        normalized_ratings[team] = normalized_value
    return normalized_ratings

def generate_elo_ratings(year):
    with open(f'./match_stats_{year}/clean_match_data_{year}.yaml', 'r') as file:
        matches = yaml.safe_load(file)

    try:
        with open('elo_ratings.yaml', 'r') as file:
            elo_ratings = yaml.safe_load(file)
            elo_ratings = normalize_ratings(elo_ratings, 1000, 1600)
            elo_ratings = {team: round(rating, 2) for team, rating in elo_ratings.items()}
    except FileNotFoundError:
        elo_ratings = {}

    rounds = defaultdict(list)
    for match in matches:
        match_id = match['MatchId']
        match_year = match_id[:4]
        round_number = match_id[4:6]
        round_key = f"{match_year}R{round_number}"
        rounds[round_key].append(match)

    sorted_round_keys = sorted(rounds.keys())

    elo_ratings_ongoing = {
        'current_ratings': {},
        'ratings_by_round': {}
    }

    for round_key in sorted_round_keys:
        round_matches = rounds[round_key]

        print(f"Processing {round_key}")
        for match in round_matches:
            team1_info = match['Team1']
            team2_info = match['Team2']
            team1_name = team1_info['Name']
            team2_name = team2_info['Name']
            team1_score = team1_info['Score']
            team2_score = team2_info['Score']

            if team1_score > team2_score:
                actual_score1, actual_score2 = 1, 0
            elif team1_score < team2_score:
                actual_score1, actual_score2 = 0, 1
            else:
                actual_score1 = actual_score2 = 0.5

            rating1 = elo_ratings.get(team1_name, 1500)
            rating2 = elo_ratings.get(team2_name, 1500)
            expected1 = expected_score(rating1, rating2)
            k = max(20, abs(team1_score - team2_score))

            new_rating1 = update_rating(rating1, expected1, actual_score1, k)
            new_rating2 = update_rating(rating2, 1 - expected1, actual_score2, k)
            elo_ratings[team1_name] = new_rating1
            elo_ratings[team2_name] = new_rating2

            print(f"{team1_name} ({team1_score}) vs {team2_name} ({team2_score})")
            print(f"Updated Ratings: {team1_name} {new_rating1:.2f}, {team2_name} {new_rating2:.2f}\n")
        rounded_ratings = {team: round(rating, 2) for team, rating in elo_ratings.items()}
        sorted_ratings = dict(sorted(rounded_ratings.items(), key=lambda x: x[1], reverse=True))
        elo_ratings_ongoing['ratings_by_round'][round_key] = sorted_ratings.copy()
    elo_ratings_ongoing['current_ratings'] = dict(sorted(rounded_ratings.items(), key=lambda x: x[1], reverse=True))
    elo_output_file = f'./match_stats_{year}/elo_ratings_{year}_ongoing.yaml'
    with open(elo_output_file, 'w') as file:
        yaml.dump(elo_ratings_ongoing, file, sort_keys=False)
