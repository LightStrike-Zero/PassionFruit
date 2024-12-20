# imports
import json
import os
from datetime import datetime

import psycopg2
# --------------------------------------------
# Database connection settings
DB_SETTINGS = {
    "dbname": "postgres",
    "user": "postgres",
    "password": "1823Black",
    "host": "localhost",
    "port": 5432,
}

# Connect to database
conn = psycopg2.connect(**DB_SETTINGS)
cursor = conn.cursor()
# --------------------------------------------
DATA_DIR = "../bulk_match_data"

# Function to convert match date to YYYY-MM-DD format
def parse_date(date_str, season_year):
    """
    Converts a date string like '14 Mar' into a valid PostgreSQL date format (YYYY-MM-DD).
    Uses the year from the season provided.
    """
    return datetime.strptime(f"{date_str} {season_year}", "%d %b %Y").strftime("%Y-%m-%d")

# Function to insert a team into the Team table
def insert_team(team_name, abbreviation):
    cursor.execute(
        """
        INSERT INTO team (team_name, abbreviation)
        VALUES (%s, %s)
        ON CONFLICT (team_name) DO NOTHING;
        """,
        (team_name, abbreviation),
    )
    cursor.execute("SELECT team_id FROM team WHERE team_name = %s;", (team_name,))
    return cursor.fetchone()[0]  # Return the team_id

# Function to insert a season into the Season table
def insert_season(season_year, num_rounds=0, teams=0):
    """
    Insert a season into the Season table or retrieve the season_id if it already exists.
    """
    cursor.execute(
        """
        INSERT INTO season (year, num_rounds, teams)
        VALUES (%s, %s, %s)
        ON CONFLICT (year) DO NOTHING; -- Avoid duplicate seasons
        """,
        (season_year, num_rounds, teams),
    )
    # Retrieve the season_id
    cursor.execute("SELECT season_id FROM season WHERE year = %s;", (season_year,))
    return cursor.fetchone()[0]  # Return the season_id

# Function to insert a match into the Match table
def insert_match(match_id, match_date, home_team, away_team, round_num, season_id):
    home_team_id = insert_team(home_team["name"], home_team["abbreviation"])
    away_team_id = insert_team(away_team["name"], away_team["abbreviation"])

    cursor.execute(
        """
        INSERT INTO match (
            match_id, match_date, home_team, away_team, round, season
        )
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (match_id) DO NOTHING;
        """,
        (match_id, match_date, home_team_id, away_team_id, round_num, season_id),
    )

def process_matches(matches, round_num, season_year, season_id):
    """
    Process match details and insert into the Match table.
    """
    match_ids = matches["MatchId"]
    match_dates = matches["MatchDate"]
    home_teams = matches["HomeTeam"]
    away_teams = matches["AwayTeam"]

    match_date_map = {}  # To map MatchId to MatchDate

    for i in range(len(match_ids)):
        match_id = match_ids[i]
        match_date = parse_date(match_dates[i], season_year)
        home_team = {"name": home_teams[i], "abbreviation": matches["HomeAbbreviation"][i]}
        away_team = {"name": away_teams[i], "abbreviation": matches["AwayAbbreviation"][i]}

        # Insert match details
        insert_match(match_id, match_date, home_team, away_team, round_num, season_id)

        # Populate match_date_map for later use
        match_date_map[match_id] = match_date

    return match_date_map

# Function to insert a player into the Player table
def insert_player(player_name):
    """
    Insert a player into the Player table or retrieve their player_id if they already exist.
    """
    cursor.execute(
        """
        INSERT INTO player (player_name)
        VALUES (%s)
        ON CONFLICT (player_name) DO NOTHING; -- Avoid duplicate players
        """,
        (player_name,),
    )
    cursor.execute("SELECT player_id FROM player WHERE player_name = %s;", (player_name,))
    return cursor.fetchone()[0]  # Return the player_id

# Function to insert player stats into the PlayerStats table
def insert_player_stats(match_id, player_id, stats):
    """
    Insert player stats into the PlayerStats table, replacing any 'NA' with a placeholder value (-99).
    """
    sanitized_stats = {key: -99 if value == "NA" else value for key, value in stats.items()}

    cursor.execute(
        """
        INSERT INTO playerstats (
            match_id, player_id, coachesvotes, ratingpoints, timeonground, kicks, handballs,
            disposals, disposalefficiency, metresgained, inside50s, contestedpossessions, 
            groundballgets, intercepts, centrebounceattendancepercentage, totalclearances,
            marks, contestedmarks, interceptmarks, shotsatgoal, goals, behinds, goalassists,
            scoreinvolvements, scorelaunches, tackles, pressureacts, hitouts
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT DO NOTHING; -- Avoid duplicate stats
        """,
        (
            match_id,
            player_id,
            sanitized_stats.get("CoachesVotes", 0),
            sanitized_stats.get("RatingPoints", 0),
            sanitized_stats.get("TimeOnGround", 0),
            sanitized_stats.get("Kicks", 0),
            sanitized_stats.get("Handballs", 0),
            sanitized_stats.get("Disposals", 0),
            sanitized_stats.get("DisposalEfficiency", 0),
            sanitized_stats.get("MetresGained", 0),
            sanitized_stats.get("Inside50s", 0),
            sanitized_stats.get("ContestedPossessions", 0),
            sanitized_stats.get("GroundBallGets", 0),
            sanitized_stats.get("Intercepts", 0),
            sanitized_stats.get("CentreBounceAttendancePercentage", 0),
            sanitized_stats.get("TotalClearances", 0),
            sanitized_stats.get("Marks", 0),
            sanitized_stats.get("ContestedMarks", 0),
            sanitized_stats.get("InterceptMarks", 0),
            sanitized_stats.get("ShotsAtGoal", 0),
            sanitized_stats.get("Goals", 0),
            sanitized_stats.get("Behinds", 0),
            sanitized_stats.get("GoalAssists", 0),
            sanitized_stats.get("ScoreInvolvements", 0),
            sanitized_stats.get("ScoreLaunches", 0),
            sanitized_stats.get("Tackles", 0),
            sanitized_stats.get("PressureActs", 0),
            sanitized_stats.get("Hitouts", 0),
        ),
    )

# Processing loop
for filename in sorted(os.listdir(DATA_DIR)):
    if filename.endswith(".json"):
        with open(os.path.join(DATA_DIR, filename), "r") as file:
            data = json.load(file)

        # Extract season and round information from Summary
        summary = data["Summary"][0]
        season_year = summary["Season"]
        round_num = summary["RoundNumber"]

        # Insert season and retrieve season_id
        season_id = insert_season(season_year)
        print(f"Processing season: {season_year} (season_id: {season_id}), round: {round_num}")

        # Process match details
        matches = data["Matches"][0]
        match_date_map = process_matches(matches, round_num, season_year, season_id)

        # Process player stats
        match_ids = data["Data"][0]["MatchId"]
        players = data["Data"][0]["Player"]
        teams = data["Data"][0]["Team"]
        stats = data["Data"][0]

        for i in range(len(match_ids)):
            match_id = match_ids[i]
            player_name = players[i]
            team_name = teams[i]

            print(f"Processing match: {match_id}, player: {player_name}, team: {team_name}")

            try:
                # Retrieve or insert team
                cursor.execute(
                    """
                    SELECT team_id FROM team WHERE team_name = %s;
                    """,
                    (team_name,),
                )
                result = cursor.fetchone()

                if result is None:
                    print(f"Team '{team_name}' not found in database. It will be inserted.")
                    team_id = insert_team(team_name, team_name[:3].upper())
                else:
                    team_id = result[0]
                    print(f"Team '{team_name}' found in database with ID {team_id}.")

                # Insert or find the player
                player_id = insert_player(player_name)
                print(f"Player '{player_name}' inserted or found with ID {player_id}.")

                # Extract player-specific stats
                player_stats = {key: stats.get(key, [0])[i] for key in stats if key not in ["MatchId", "Player", "Team"]}
                print(f"Player stats for player '{player_name}': {player_stats}")

                # Insert player stats
                insert_player_stats(match_id, player_id, player_stats)

            except Exception as e:
                print(f"Error processing match: {match_id}, player: {player_name}, team: {team_name}")
                print(f"Error details: {e}")
                raise

# Commit and close connection
conn.commit()
cursor.close()
conn.close()

print("Data inserted successfully!")