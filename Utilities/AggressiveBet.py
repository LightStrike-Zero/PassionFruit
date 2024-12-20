import json
import yaml

with open('AFL_Betting_Data_2024.json', 'r') as json_file:
    json_data = json.load(json_file)

with open('../match_stats_2024/clean_match_data_2024.yaml', 'r') as yaml_file:
    yaml_data = yaml.safe_load(yaml_file)


bet_amount = 1.0

json_dict = {entry['MatchId']: entry for entry in json_data}

total_outlay = 0.0
total_won = 0.0
total_lost = 0.0
win_count = 0
loss_count = 0

for match in yaml_data:
    match_id = int(match['MatchId'])
    team1 = match['Team1']
    team2 = match['Team2']

    json_match = json_dict.get(match_id)
    if not json_match:
        print(f"MatchId {match_id} not found in JSON data")
        continue

    total_outlay += bet_amount

    if json_match['Home Odds'] > json_match['Away Odds']:
        bet_team = json_match['Home Team']
        bet_odds = json_match['Home Odds']
    else:
        bet_team = json_match['Away Team']
        bet_odds = json_match['Away Odds']

    if team1['Result'] == 'W':
        winning_team = team1['Name']
    elif team2['Result'] == 'W':
        winning_team = team2['Name']
    else:
        total_lost += bet_amount  # Lost bet on a drawn match
        loss_count += 1
        print(f"MatchId {match_id}: Drawn match - Bet lost")
        continue

    if bet_team == winning_team:
        winnings = bet_amount * bet_odds
        total_won += winnings
        win_count += 1
        print(f"MatchId {match_id}: Bet Team {bet_team}, Odds: {bet_odds}, Winnings: {winnings:.2f} - WIN")
    else:
        total_lost += bet_amount * bet_odds
        loss_count += 1
        print(f"MatchId {match_id}: Bet Team {bet_team}, Odds: {bet_odds} - Bet lost")

net_result = total_won - total_lost

print("\nFinal Results for Betting on Higher Odds")
print(f"Total Outlay (Total Bet Amount): {total_outlay:.2f}")
print(f"Total Won: {total_won:.2f}")
print(f"Total Lost: {total_lost:.2f}")
print(f"Net Result (Total Won - Total Lost): {net_result:.2f}")
print(f"Total Wins: {win_count}, Total Losses: {loss_count}")
print(f"Win Rate: {(win_count / (win_count + loss_count)) * 100:.2f}%")
