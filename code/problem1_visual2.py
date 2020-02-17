# Single Play.
import numpy as np
import cv2
import csv
import matplotlib.pyplot as plt
plt.rc('font', family='Times New Roman')

# Hyper parameters.
TIME_GAP = 600

# Read the csv file.
csv_reader = csv.reader(open('./data/passingevents.csv'))

# The first match.(First match and self passing only.)
passing_list = [row for row in csv_reader if row[0]
                == '1' and row[1] == 'Huskies']
# Fix the time of 2H.
for p in passing_list:
    if p[4] == '2H':
        p[5] = str(float(p[5])+2700)
passing_cnt = len(passing_list)

# Count the player's average pos in a single play.
player_map = {}

t = 0
pass_i = 0


def add_to_player_map(player: str, x, y):
    '''
    A function to add the position to the player.
    '''
    if player_map.get(player) is None:
        player_map[player] = {'x': x, 'y': y, 'cnt': 1}
    else:
        player_map[player]['x'] += x
        player_map[player]['y'] += y
        player_map[player]['cnt'] += 1


center_x = []
center_y = []
ds = []
spd = []

while pass_i < len(passing_list):
    t += TIME_GAP
    dx_sum = 0.0
    dy_sum = 0.0
    player_map.clear()
    while pass_i < len(passing_list) and float(passing_list[pass_i][5]) < t:
        cur_pass = passing_list[pass_i]
        add_to_player_map(cur_pass[2], float(cur_pass[7]), float(cur_pass[8]))
        add_to_player_map(cur_pass[3], float(cur_pass[9]), float(cur_pass[10]))

        dx_sum += abs(float(cur_pass[9]) - float(cur_pass[7]))
        dy_sum += abs(float(cur_pass[10]) - float(cur_pass[8]))
        pass_i += 1

    # Caculate the center x.
    x_sum = 0
    y_sum = 0
    d_sum = 0

    for k, v in player_map.items():
        x_sum += v['x'] / v['cnt']
        y_sum += v['y'] / v['cnt']
    center_x.append(x_sum / len(player_map))
    center_y.append(y_sum / len(player_map))

    for k, v in player_map.items():
        d_sum += (v['x'] / v['cnt'] - center_x[-1]) ** 2 + \
            (v['y'] / v['cnt'] - center_y[-1]) ** 2
    ds.append(d_sum / len(player_map))
    spd.append(dx_sum / dy_sum)

# Plot
plt.plot(center_x, color='blue', linewidth=1.0)
plt.ylim((0, 100))
plt.xlabel('t(10min)')
plt.ylabel('x')
plt.title('<X> pos')
plt.show()

plt.plot(center_y, color='blue', linewidth=1.0)
plt.ylim((0, 100))
plt.xlabel('time(10min)')
plt.ylabel('y')
plt.title('<Y> pos')
plt.show()

plt.plot(ds, color='blue', linewidth=1.0)
plt.ylim((0, 1300))
plt.xlabel('time(10min)')
plt.ylabel('ds')
plt.title('D')
plt.show()

plt.plot(spd, color='blue', linewidth=1.0)
plt.ylim((0, 0.8))
plt.xlabel('time(10min)')
plt.ylabel('speed')
plt.title('Speed')
plt.show()
