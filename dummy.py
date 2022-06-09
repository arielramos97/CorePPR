import numpy as np
from scipy.signal import savgol_filter
from kneed import KneeLocator
import matplotlib.pyplot as plt


ignore_n =2
y = [0.4627654552459717, 0.09401446580886841, 0.03227425739169121, 0.004365486092865467, 0.008818025700747967, 0.00828575063496828, 0.0044616456143558025, 0.009923656471073627, 0.011287947185337543, 0.008227081038057804, 0.009971088729798794, 0.0101219043135643, 0.03525698557496071, 0.0022258898243308067, 0.027026163414120674, 0.08676853030920029, 3.8536258216481656e-05, 0.08676855266094208, 0.08676853030920029, 5.7566114264773205e-05]
# x = range(0, len(y))
x = np.arange(len(y) - ignore_n)

# print('x: ', len(x))
y = np.array(y)
idx_y = np.argsort(y)[::-1]

y = y[idx_y]

# print('y: ' ,len(y))

y = y[ignore_n:]
# print('new y: ' ,len(y))

half_length = int(len(y) * 0.10)


if half_length % 2 == 0:
    window = half_length + 1
else:
    window = half_length

print('window size: ', window)

if window <= 1:
    window = 3

smoothed_y = savgol_filter(y, window, 1)

sensitivity = 1


kn = KneeLocator(x, y, curve='convex', direction='decreasing', S=sensitivity)
print('kn: ', kn.knee)

kn_smooth = KneeLocator(x, smoothed_y, curve='convex', direction='decreasing', S=sensitivity)
print('smoothed kn: ', kn_smooth.knee)


fig, (ax1, ax2) = plt.subplots(2)


ax2.set_xlabel('k')
ax2.set_ylabel('PageRank values')
ax2.plot(x, smoothed_y, 'rx-')
ax2.vlines(kn_smooth.knee, plt.ylim()[0], plt.ylim()[1], linestyles='dotted')

ax1.set_xlabel('k')
ax1.set_ylabel('PageRank values')
ax1.plot(x, y, 'bx-')
ax1.vlines(kn.knee, plt.ylim()[0], plt.ylim()[1], linestyles='dashed')



plt.show()
