import json
import yaml
import random

with open('AFL_Betting_Data_2024.json', 'r') as json_file:
    json_data = json.load(json_file)

with open('../match_stats_2024/clean_match_data_2024.yaml', 'r') as yaml_file:
    yaml_data = yaml.safe_load(yaml_file)

bet_amount = 1.0

json_dict = {entry['MatchId']: entry for entry in json_data}

num_simulations = 100

total_outlay_accum = 0.0
total_won_accum = 0.0
total_lost_accum = 0.0
net_result_accum = 0.0
win_count_accum = 0
loss_count_accum = 0

for sim in range(num_simulations):
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
            continue

        total_outlay += bet_amount

        if random.choice([True, False]):
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
            total_lost += bet_amount
            loss_count += 1
            continue

        if bet_team == winning_team:
            winnings = bet_amount * bet_odds
            total_won += winnings
            win_count += 1
        else:
            total_lost += bet_amount * bet_odds
            loss_count += 1

    net_result = total_won - total_lost

    total_outlay_accum += total_outlay
    total_won_accum += total_won
    total_lost_accum += total_lost
    net_result_accum += net_result
    win_count_accum += win_count
    loss_count_accum += loss_count

    print(f"\nSimulation {sim + 1} Results")
    print(f"Total Outlay (Total Bet Amount): {total_outlay:.2f}")
    print(f"Total Won: {total_won:.2f}")
    print(f"Total Lost: {total_lost:.2f}")
    print(f"Net Result (Total Won - Total Lost): {net_result:.2f}")
    print(f"Total Wins: {win_count}, Total Losses: {loss_count}")
    if win_count + loss_count > 0:
        print(f"Win Rate: {(win_count / (win_count + loss_count)) * 100:.2f}%")

avg_outlay = total_outlay_accum / num_simulations
avg_won = total_won_accum / num_simulations
avg_lost = total_lost_accum / num_simulations
avg_net_result = net_result_accum / num_simulations
avg_win_rate = (win_count_accum / (win_count_accum + loss_count_accum)) * 100 if (win_count_accum + loss_count_accum) > 0 else 0

print(f"\nAverage Results Over {num_simulations} Simulations")
print(f"Average Total Outlay: {avg_outlay:.2f}")
print(f"Average Total Won: {avg_won:.2f}")
print(f"Average Total Lost: {avg_lost:.2f}")
print(f"Average Net Result (Total Won - Total Lost): {avg_net_result:.2f}")
print(f"Average Wins: {win_count_accum // num_simulations}, Average Losses: {loss_count_accum // num_simulations}")
print(f"Average Win Rate: {avg_win_rate:.2f}%")
