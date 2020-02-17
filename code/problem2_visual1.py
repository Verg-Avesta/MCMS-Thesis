import numpy as np
import cv2
import csv
import matplotlib.pyplot as plt
plt.rc('font', family='Times New Roman')

# Hyper Parameters
POINT_NUM = 10

# Read the csv file.
csv_reader = list(csv.reader(open('./data/passingevents.csv')))[1:]

# The first match.(First match and self passing only.)
passing_list = [row for row in csv_reader]
passing_cnt = len(passing_list)

# Set the x-range.
x = np.linspace(0, 100, POINT_NUM)
y = [[] for _ in range(POINT_NUM)]
dis_h = []
dis_o = []
pass_y = np.zeros(POINT_NUM)
pass_longx_h = np.zeros(POINT_NUM)
pass_longx_o = np.zeros(POINT_NUM)

for i in range(passing_cnt - 1):
    distance = ((float(passing_list[i][9]) - float(passing_list[i][7])) **
                2 + (float(passing_list[i][10]) - float(passing_list[i][8])) ** 2) ** 0.5
    distance += ((float(passing_list[i+1][7]) - float(passing_list[i][7])) **
                 2 + (float(passing_list[i + 1][8]) - float(passing_list[i][8])) ** 2) ** 0.5
    t = float(passing_list[i + 1][5]) - float(passing_list[i][5])

    if t <= 0.2:
        continue

    average_x = (float(
        passing_list[i][7]) + float(passing_list[i][9]) + float(passing_list[i + 1][7])) / 3
    average_y = (float(
        passing_list[i][8]) + float(passing_list[i][10])) / 2

    y[int(average_x * POINT_NUM / 100)].append(distance / t)
    pass_y[int(average_x * POINT_NUM / 100)] += 1

    if float(passing_list[i][7]) > 0:
        if passing_list[i][1] == 'Huskies':
            pass_longx_h[int(average_y * POINT_NUM / 100)] += 1
            dis_h.append(distance)
        else:
            pass_longx_o[int(average_y * POINT_NUM / 100)] += 1
            dis_o.append(distance)


y = np.array([(np.mean(np.array(i)) if len(i) > 0 else np.nan) for i in y])

# plt.plot(x, y, color='blue', linewidth=1)
# plt.xlabel('x-pos')
# plt.ylabel('average speed')
# plt.show()

# plt.plot(x, pass_y, color='blue', linewidth=1)
# plt.xlabel('x-pos')
# plt.ylabel('Passing Count')
# plt.show()

# pass_longx_h /= np.sum(pass_longx_h)
# pass_longx_o /= np.sum(pass_longx_o)

# plt.plot(x, pass_longx_h, color='blue', linewidth=1, label='Huskies')
# plt.plot(x, pass_longx_o, color='red', linewidth=1, label='Opponents')
# plt.legend()
# plt.xlabel('y-pos')
# plt.ylabel('Passing Count')
# plt.show()

plt.hist(dis_o, bins=50, color='yellow', label='Opponents', alpha=0.3)
plt.hist(dis_h, bins=50, color='purple', label='Huskies', alpha=0.3)

plt.legend()
plt.show()
