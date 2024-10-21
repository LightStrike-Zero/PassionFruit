import yaml

# Load the YAML file
def load_yaml(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data

# Extract schedule information
def extract_schedule(data):
    schedule = []

    for match in data:
        match_id = match['MatchId']
        team1_name = match['Team1']['Name']
        team2_name = match['Team2']['Name']

        schedule_entry = {
            'MatchId': match_id,
            'Team1': team1_name,
            'Team2': team2_name
        }

        schedule.append(schedule_entry)

    return schedule

# Save the schedule to a new YAML file
def save_schedule_to_yaml(schedule, output_file):
    with open(output_file, 'w') as file:
        yaml.dump(schedule, file)

# Main function
def main(input_yaml, output_yaml):
    # Load the YAML match data
    data = load_yaml(input_yaml)

    # Extract schedule
    schedule = extract_schedule(data)

    # Save the schedule to a new YAML file
    save_schedule_to_yaml(schedule, output_yaml)

    # Print a success message
    print(f"Schedule saved to: {output_yaml}")

year = '2024'

# Replace with the actual path to your input and output YAML files
input_yaml = f'match_stats_{year}/clean_match_data_{year}.yaml'
output_yaml = f'match_stats_{year}/match_schedule_{year}.yaml'

# Run the main function
main(input_yaml, output_yaml)
