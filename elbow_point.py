import numpy as np
import math
import matplotlib.pyplot as plt


def get_elbow_point2(sorted_scores):

    if (len(sorted_scores)<2):
        elbow_point = 1
        return elbow_point
    else:  

        first_point = np.zeros((2), dtype=np.float32)
        first_point[0] = 0
        first_point[1] = sorted_scores[0]

        last_point = np.zeros((2), dtype=np.float32)
        last_point[0] = len(sorted_scores)-1
        last_point[1] = sorted_scores[-1]


        k =1
        distances = np.zeros((len(sorted_scores)), dtype=np.float32)

        for i in range(len(sorted_scores)):
            numerator = abs((last_point[1] - first_point[1])*k - (last_point[0] - first_point[0])*sorted_scores[i] + last_point[0]*first_point[1] - last_point[1]*first_point[0])
            denominator = (pow((last_point[1] - first_point[1]),2) + pow(pow((last_point[0] - first_point[0]),2),0.5))
            distances[i] = numerator/denominator
            k = k + 1

        elbow_point = np.argmax(distances)
        return elbow_point


def get_elbow_point(sorted_scores):

    if (len(sorted_scores)<2):
        elbow_point = 1
        return elbow_point
    else:

        first_point = (0, sorted_scores[0])
        last_point = (len(sorted_scores)-1, sorted_scores[-1])

        k =1
        distances = []

        for point in sorted_scores:
            numerator = abs((last_point[1] - first_point[1])*k - (last_point[0] - first_point[0])*point + last_point[0]*first_point[1] - last_point[1]*first_point[0])
            denominator = (pow((last_point[1] - first_point[1]),2) + pow(pow((last_point[0] - first_point[0]),2),0.5))
            distances.append(numerator/denominator)
            k = k + 1

        distances_array = np.array(distances)

        candidate_elbow_points = np.where(distances_array==np.max(distances_array))

        # print(candidate_elbow_points)

        if len(candidate_elbow_points)>=2:
            elbow_point = candidate_elbow_points[0]
        else:
            elbow_point = candidate_elbow_points

        
        return elbow_point[0][0]


