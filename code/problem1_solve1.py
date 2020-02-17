import numpy as np
import cv2
import csv

# Hyper Parameters
L = 2

# Read the csv file.
csv_reader = csv.reader(open('./data/passingevents.csv'))

# The first match.(First match and self passing only.)
passing_list = [row for row in csv_reader if row[1] == 'Huskies']
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
pass_data = []

# Count the passing cnt.
for row in passing_list:
    if len(pass_data) == 0 or pass_data[-1][-1] != row[2]:
        pass_data.append([row[2], row[3]])
    else:
        pass_data[-1].append(row[3])

# Find the most frequent methods.
pass_map = {}

for long_pass in pass_data:
    if len(long_pass) < L:
        continue
    for i in range(len(long_pass) - L + 1):
        cur_cnt = 0
        cur_pass = {}
        cur_ans = ''
        for j in range(i, len(long_pass) - L + 1):
            if cur_pass.get(long_pass[j]) is None:
                cur_ans += str(cur_cnt) + '-'
                cur_pass[long_pass[j]] = cur_cnt
                cur_cnt += 1
            else:
                cur_ans += str(cur_pass[long_pass[j]]) + '-'

            if j == len(long_pass) - L + 1 and cur_cnt == L:
                cur_ans = cur_ans[:-1]
                if pass_map.get(cur_ans) is None:
                    pass_map[cur_ans] = 1
                else:
                    pass_map[cur_ans] += 1
            elif cur_cnt > L:
                cur_ans = cur_ans[:-3]
                if pass_map.get(cur_ans) is None:
                    pass_map[cur_ans] = 1
                else:
                    pass_map[cur_ans] += 1
                break

sorted_list = sorted(pass_map.items(), key=lambda x: x[1], reverse=True)

for item in sorted_list:
    print(item)
