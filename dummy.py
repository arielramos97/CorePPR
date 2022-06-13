import numpy as np
from scipy.signal import savgol_filter
from kneed import KneeLocator
import matplotlib.pyplot as plt


ignore_n =0
y = [0.5, 0.005319148767739534, 0.005376965738832951, 0.005436709616333246, 0.005487049464136362, 0.005528618115931749, 0.005582820158451796, 0.0056326668709516525, 0.005693891551345587, 0.005757157225161791, 0.0057895006611943245, 0.0058441185392439365]

# x = range(0, len(y))
x = np.arange(len(y) - ignore_n)

# print('x: ', len(x))
y = np.array(y)
idx_y = np.argsort(y)[::-1]

y = y[idx_y]

# print('y: ' ,len(y))

y = y[ignore_n:]
# print('new y: ' ,len(y))

half_length = 3 # int(len(y)/2)


if half_length % 2 == 0:
    window = half_length + 1
else:
    window = half_length

print('window size: ', window)

if window <= 1:
    window = 3


# smoothed_y = savgol_filter(y, window, 1)

sensitivity = 1


kn = KneeLocator(x, y, curve='convex', direction='decreasing', S=sensitivity)
print('kn: ', kn.knee)

# kn_smooth = KneeLocator(x, smoothed_y, curve='convex', direction='decreasing', S=sensitivity)
# print('smoothed kn: ', kn_smooth.knee)


fig, (ax1, ax2) = plt.subplots(2)


# ax2.set_xlabel('k')
# ax2.set_ylabel('PageRank values')
# ax2.plot(x, smoothed_y, 'rx-')
# ax2.vlines(kn_smooth.knee, plt.ylim()[0], plt.ylim()[1], linestyles='dotted')

ax1.set_xlabel('k')
ax1.set_ylabel('PageRank values')
ax1.plot(x, y, 'bx-')
ax1.vlines(kn.knee, plt.ylim()[0], plt.ylim()[1], linestyles='dashed')



plt.show()
