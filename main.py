
import sys

import pandas as pd
import random
import numpy as np
import math
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import style


# decided things

known_rows = 500  # Number of known points of specific categories to find. Errors if this value is too large.
testing_rows = 100  # Number of unknown points of specific categories to find.
all_groups = [["DRUG/NARCOTIC", 'lime'], ["MISSING PERSON", 'orange'], ["VEHICLE THEFT", 'blue'], ["BURGLARY", 'magenta']]  # [["category', 'color'], ...]
cutoff_point = 50000  # row number in dataset where the known points transition to the unknown points

# ["ASSAULT", 'red'], ["DRUG/NARCOTIC", 'green'], ["MISSING PERSON", 'orange'], ["VEHICLE THEFT", 'blue'], ["BURGLARY", 'magenta']

# getting the data

data_file = pd.read_csv("sanfransisco_crime.csv", header=0, nrows=cutoff_point)

skipping_rows = []
for i in range(1, cutoff_point+1):
    skipping_rows.append(i)
testing_file = pd.read_csv("sanfransisco_crime.csv", header=0, skiprows=skipping_rows)


all_group_names = []
for group in all_groups:
    all_group_names.append(group[0])
    print(group)

group = 0

# setting the plot background as a map of san fran
img = plt.imread("sanfran_map.png")
fig, ax = plt.subplots()
ax.imshow(img, extent=(-122.5423, -122.332, 37.6871, 37.824))
# plt.plot(-122.477163, 37.811053, 'o', color='black', markersize=5)
# plt.plot(-122.359528, 37.718874, 'o', color='black', markersize=5)


def groupPoint(grouped_points_k, point_u, k):
    # grouped_points_k = the 2D array of known points
    # point_u = the unknown point that whose point must be decided
    # k = the number of nearest points to consider when deciding the group of point_u

    # returns 0 or 1 based on what group it decides point_u is in
    # all point objects must be in the following format:  [x position, y position, category]

    # getting average distances
    average_time_dist = 0
    average_euclidean_dist = 0
    for point in grouped_points_k:
        # time distance
        lower_time = min(point[3], point_u[3])
        higher_time = max(point[3], point_u[3])
        average_time_dist += min(higher_time - lower_time, 24 - (higher_time - lower_time))
        # euclidean distance
        average_euclidean_dist += math.sqrt(((point[0]-point_u[0])**2)+((point[1]-point_u[1])**2))
    average_time_dist /= known_rows
    average_euclidean_dist /= known_rows

    # calculating distances
    distance_and_group = []
    for point in grouped_points_k:

        lower_time = min(point[3], point_u[3])
        higher_time = max(point[3], point_u[3])
        time_dist = min(higher_time - lower_time, 24 - (higher_time - lower_time))

        euclidean_dist = math.sqrt(((point[0]-point_u[0])**2)+((point[1]-point_u[1])**2))

        normalized_euclidean_dist = euclidean_dist*(10/average_euclidean_dist)
        normalized_time_dist = time_dist*(1/average_time_dist)
        total_dist = normalized_euclidean_dist + normalized_time_dist

        distance_and_group.append([total_dist, point[2]])

    distance_and_group = sorted(distance_and_group)
    k_nearest_points = distance_and_group[:k]


    group_scores = []
    for group in all_groups:
        group_scores.append(0)

    for chosen_point in k_nearest_points:
        group_scores[all_group_names.index(chosen_point[1])] += 1

    return all_group_names[group_scores.index(max(group_scores))]


def main():

    # getting the known points from the cvs file
    known_points = []
    i_again = 0

    while len(known_points) < known_rows:

        for group in all_groups:
            if data_file.at[i_again, 'Category'] == group[0]:

                time_int = (int(data_file.at[i_again, 'Time'][0]) * 10) + int(data_file.at[i_again, 'Time'][1]) + (
                            int(data_file.at[i_again, 'Time'][3]) / 6) + (int(data_file.at[i_again, 'Time'][4]) / 60)

                known_points.append([data_file.at[i_again, 'X'], data_file.at[i_again, 'Y'], data_file.at[i_again, 'Category'], time_int])

                x = data_file.at[i_again, 'X']
                y = data_file.at[i_again, 'Y']

                plt.plot(x, y, 'o', color=group[1], markersize=4)

        i_again += 1

    # getting the unknown points from the csv file
    unknown_points = []
    unknown_points_rows = []
    i_this = 0
    while i_this < testing_rows:
        number_of_unknown_rows = int((testing_file.size / testing_file.columns.size)) - 1
        random_row = random.randint(0, number_of_unknown_rows)

        for group in all_groups:
            if testing_file.at[random_row, 'Category'] == group[0]:

                time_int = (int(testing_file.at[random_row, 'Time'][0]) * 10) + int(testing_file.at[random_row, 'Time'][1]) + (
                            int(testing_file.at[random_row, 'Time'][3]) / 6) + (int(testing_file.at[random_row, 'Time'][4]) / 60)

                unknown_points.append([testing_file.at[random_row, 'X'], testing_file.at[random_row, 'Y'], testing_file.at[random_row, 'Category'], time_int])
                unknown_points_rows.append(random_row)
                i_this += 1

    # running algorithm / measuring accuracy

    total_correct = 0
    for unknown_point_i in unknown_points:

        output = groupPoint(known_points, unknown_point_i, 5)

        x = unknown_point_i[0]
        y = unknown_point_i[1]
        plt.plot(x, y, 'x', color=all_groups[all_group_names.index(output)][1], markersize=8)

        if output == unknown_point_i[2]:
            total_correct += 1

    print("Percent Accuracy:", str((total_correct/testing_rows)*100), "%")
    print("Expected Accuracy if it were just guessing randomly:", str(100/len(all_groups)), "%")




    # plotting

    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()


if __name__ == '__main__':
    main()
# :)
