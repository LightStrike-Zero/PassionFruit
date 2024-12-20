import yaml
from collections import defaultdict

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

# Load match data from YAML file
with open('../match_stats_2023/clean_match_data_2023.yaml', 'r') as file:
    matches = yaml.safe_load(file)

# Load or initialize Elo ratings
try:
    with open('elo_ratings.yaml', 'r') as file:
        elo_ratings = yaml.safe_load(file)
        # Normalize the loaded ratings
        elo_ratings = normalize_ratings(elo_ratings, 1000, 1600)
        # Round the normalized ratings
        elo_ratings = {team: round(rating, 2) for team, rating in elo_ratings.items()}
except FileNotFoundError:
    # Initialize elo_ratings to an empty dict
    elo_ratings = {}

# Build a dictionary of rounds
rounds = defaultdict(list)
for match in matches:
    match_id = match['MatchId']
    year = match_id[:4]
    round_number = match_id[4:6]
    round_key = f"{year}R{round_number}"
    rounds[round_key].append(match)

# Get the sorted list of round keys
sorted_round_keys = sorted(rounds.keys())

# Initialize the ongoing Elo ratings data structure
elo_ratings_ongoing = {
    'current_ratings': {},
    'ratings_by_round': {}
}

# Process each round
for round_key in sorted_round_keys:
    round_matches = rounds[round_key]

    print(f"Processing {round_key}")
    # Process matches in this round
    for match in round_matches:
        team1_info = match['Team1']
        team2_info = match['Team2']

        team1_name = team1_info['Name']
        team2_name = team2_info['Name']

        team1_score = team1_info['Score']
        team2_score = team2_info['Score']

        # Determine match result
        if team1_score > team2_score:
            team1_result = 'W'
            team2_result = 'L'
            actual_score1 = 1
            actual_score2 = 0
        elif team1_score < team2_score:
            team1_result = 'L'
            team2_result = 'W'
            actual_score1 = 0
            actual_score2 = 1
        else:
            team1_result = 'D'
            team2_result = 'D'
            actual_score1 = 0.5
            actual_score2 = 0.5

        # Get current Elo ratings or initialize them
        rating1 = elo_ratings.get(team1_name, 1500)
        rating2 = elo_ratings.get(team2_name, 1500)

        # Calculate expected scores
        expected1 = expected_score(rating1, rating2)
        expected2 = 1 - expected1  # Alternatively, recalculate using expected_score(rating2, rating1)

        # Calculate K as the absolute difference in scores, minimum of 20
        k = max(30, abs(team1_score - team2_score))

        # Update ratings
        new_rating1 = update_rating(rating1, expected1, actual_score1, k)
        new_rating2 = update_rating(rating2, expected2, actual_score2, k)

        # Store updated ratings
        elo_ratings[team1_name] = new_rating1
        elo_ratings[team2_name] = new_rating2

        # Print match results and updated ratings
        print(f"{team1_name} ({team1_score}) vs {team2_name} ({team2_score})")
        print(f"Result: {team1_result}-{team2_result}")
        print(f"Expected: {team1_name} {expected1:.3f}, {team2_name} {expected2:.3f}")
        print(f"K-value: {k}")
        print(f"Updated Ratings: {team1_name} {new_rating1:.2f}, {team2_name} {new_rating2:.2f}\n")

    # After processing all matches in the round, store the ratings after this round
    # Round the ratings to two decimal places before storing
    rounded_ratings = {team: round(rating, 2) for team, rating in elo_ratings.items()}

    # Sort the ratings by rating in descending order
    sorted_ratings = dict(sorted(rounded_ratings.items(), key=lambda x: x[1], reverse=True))

    # Store the sorted ratings in 'ratings_by_round'
    elo_ratings_ongoing['ratings_by_round'][round_key] = sorted_ratings.copy()

# After processing all rounds, set the current_ratings
# Also sort the current_ratings by Elo descending
elo_ratings_ongoing['current_ratings'] = dict(sorted(rounded_ratings.items(), key=lambda x: x[1], reverse=True))

# Save updated ratings to a separate file
with open('elo_ratings_ongoing.yaml', 'w') as file:
    yaml.dump(elo_ratings_ongoing, file, sort_keys=False)
