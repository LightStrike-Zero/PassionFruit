import yaml

# Load Elo ratings from the YAML file
with open('elo_ratings.yaml', 'r') as file:
    elo_ratings = yaml.safe_load(file)

# Convert ratings to a list of tuples (team, rating), rounding ratings
elo_list = [(team, round(rating, 2)) for team, rating in elo_ratings.items()]

# Sort the list by rating in descending order
sorted_elo = sorted(elo_list, key=lambda x: x[1], reverse=True)

# Create a new dictionary that maintains order (Python 3.7+)
sorted_elo_dict = dict(sorted_elo)

# Print the sorted ratings
print("Elo Ratings from Highest to Lowest:")
for rank, (team, rating) in enumerate(sorted_elo, start=1):
    print(f"{rank}. {team}: {rating:.2f}")

# Write the sorted Elo ratings to a YAML file
with open('sorted_elo_ratings.yaml', 'w') as file:
    yaml.dump(sorted_elo_dict, file, sort_keys=False)
