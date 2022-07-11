import numpy as np
import math
import matplotlib.pyplot as plt



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






# test = [0.00836235668257986, 0.009554730797469947, 0.009594503232387227, 0.009660828320889355, 0.010365080136547463, 0.011133399782425687, 0.012249794127531471, 0.012478599859255252, 0.012904704557203665, 0.013237280328251623, 0.014886060211458435, 0.01605162174605241, 0.018034599890199583, 0.019871486454287837, 0.02086771348869478, 0.02225295322334127, 0.02475311073696887, 0.025596922220033062, 0.025680762720780057, 0.03228773855973504, 0.03980941564811553, 0.04696287234911033, 0.30575710573073317]

# ignore = 1

# x = np.arange(len(test)-ignore)
# test = np.array(test)
# test_sorted = np.sort(test)[::-1]

# test_sorted   = test_sorted[ignore:]

# print(test_sorted)

# e_point = get_elbow_point(test_sorted.tolist())
# print(e_point)


# fig, (ax1, ax2) = plt.subplots(2)


# # ax2.set_xlabel('k')
# # ax2.set_ylabel('PageRank values')
# # ax2.plot(x, smoothed_y, 'rx-')
# # ax2.vlines(kn_smooth.knee, plt.ylim()[0], plt.ylim()[1], linestyles='dotted')



# ax1.set_xlabel('k')
# ax1.set_ylabel('PageRank values')
# ax1.plot(x, test_sorted, 'bx-')
# ax1.vlines(e_point, np.min(test_sorted), np.max(test_sorted), linestyles='dashed')




# plt.show()
