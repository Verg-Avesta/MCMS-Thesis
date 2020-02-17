import numpy as np
import cv2
import csv

# Read the csv file.
csv_reader = csv.reader(open('./data/passingevents.csv'))

# The first match.(First match and self passing only.)
passing_list = [row for row in csv_reader if row[0]
                == '1' and row[1] == 'Huskies' and row[4] == '1H']
passing_cnt = len(passing_list)

# Analyzing the data.
player_map = {}
player_list = []
for row in passing_list:
    if player_map.get(row[2]) is None:
        player_map[row[2]] = len(player_list)
        player_list.append(row[2])
    if player_map.get(row[3]) is None:
        player_map[row[3]] = len(player_list)
        player_list.append(row[3])

player_cnt = len(player_list)
passing_matrix: np.ndarray = np.zeros((player_cnt, player_cnt))

# Count the passing cnt.
for row in passing_list:
    origin_id = player_map[row[2]]
    dest_id = player_map[row[3]]

    # Update the passing matrix.
    passing_matrix[origin_id][dest_id] += 1

for i in range(player_cnt):
    for j in range(player_cnt):
        if i == j:
            continue
        print(
            f'''['{player_list[i][-2:]}','{player_list[j][-2:]}',{passing_matrix[i][j]}],''')
