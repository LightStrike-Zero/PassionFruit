import os
import requests


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


# Usage
year = 2022  # Specify the year
rounds = 24  # Number of rounds
folder_name = f'match_stats_{year}'  # Folder to save the JSON files

download_json_files(year, rounds, folder_name)
