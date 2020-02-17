# The passing network in a play.
import numpy as np
import cv2
import csv

# Hyper parameters.
W = 1000
H = 500
LINE_COLOR_UB = (150, 60, 100)
POINT_COLOR_UB = (0, 255, 255)
LINE_COLOR_LB = (255, 180, 160)
POINT_COLOR_LB = (0, 0, 200)
# Blue: F, Green: D, Purple: M, Yellow: G.
PLAYER_COLOR = {'F': (0xff, 0x99, 0x66), 'D': (0x66, 0x99, 0x66),
                'M': (0x66, 0x33, 0x33), 'G': (0, 0xff, 0xff)}

LINE_WIDTH_LB = 1
LINE_WIDTH_UB = 10
POINT_RADIUS = 20
PLAYER_BORDER = 3

# Read the csv file.
csv_reader = csv.reader(open('./data/passingevents.csv'))

# The first match.(First match and self passing only.)
passing_list = [row for row in csv_reader if row[0]
                == '26' and row[1] != 'Huskies' and row[4] == '1H']

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
player_pos = np.zeros((player_cnt, 2))
player_passing_cnt = np.zeros(player_cnt)

# Count the passing cnt.
for row in passing_list:
    origin_id = player_map[row[2]]
    dest_id = player_map[row[3]]

    # Update the passing matrix.
    passing_matrix[origin_id][dest_id] += 1
    passing_matrix[dest_id][origin_id] += 1

    # Update the player's position.
    player_pos[origin_id][0] += float(row[7])
    player_pos[origin_id][1] += float(row[8])
    player_pos[dest_id][0] += float(row[9])
    player_pos[dest_id][1] += float(row[10])

    player_passing_cnt[origin_id] += 1
    player_passing_cnt[dest_id] += 1

for pos in range(player_cnt):
    if player_passing_cnt[pos] != 0:
        player_pos[pos][0] /= player_passing_cnt[pos]
        player_pos[pos][1] /= player_passing_cnt[pos]


def pos_transforms(origin_pos):
    '''
    A function to transforms the position from the raw data to the opencv image.
    '''
    return int(origin_pos[0]/100*W), int((100-origin_pos[1])/100*H)


def color_transforms(origin_color, dest_color, rate):
    return_color = []
    for s, d in zip(origin_color, dest_color):
        return_color.append(int(s + (d - s) * rate))
    return tuple(return_color)


# Create the figure.
img = cv2.imread('./data/football_background.png')
W = img.shape[1]
H = img.shape[0]

# Paint the lines.
max_passing = np.max(passing_matrix)
max_player = np.max(player_passing_cnt)
min_passing = np.min(passing_matrix)
min_player = np.min(passing_matrix)

for origin in range(player_cnt):
    for dest in range(player_cnt):
        lw = LINE_WIDTH_LB + (passing_matrix[origin][dest] - min_passing) / (max_passing - min_passing) * (
            LINE_WIDTH_UB - LINE_WIDTH_LB) if passing_matrix[origin][dest] != 0 else 0

        if lw > 0:
            color_rate = (passing_matrix[origin][dest] -
                          min_passing) / (max_passing - min_passing)
            cv2.line(img, pos_transforms(player_pos[origin]), pos_transforms(
                player_pos[dest]), color_transforms(LINE_COLOR_LB, LINE_COLOR_UB, color_rate), int(lw), cv2.LINE_AA)

# Paint the points.
for i in range(player_cnt):
    color_rate = (1-(player_passing_cnt[i] - min_player) /
                  (max_player - min_player))
    radius = int((1 - (1 - player_passing_cnt[i] /
                       (max_player - min_player)) * 0.65) * POINT_RADIUS)
    border_color = PLAYER_COLOR[player_list[i][-2]]
    cv2.circle(img, pos_transforms(player_pos[i]), radius + PLAYER_BORDER,
               border_color, -1, cv2.LINE_AA)
    cv2.circle(img, pos_transforms(player_pos[i]), radius,
               color_transforms(POINT_COLOR_LB, POINT_COLOR_UB, color_rate), -1, cv2.LINE_AA)

cv2.imshow('Problem1', img)
cv2.waitKey(0)

cv2.imwrite('problem1_visual1.png', img)
