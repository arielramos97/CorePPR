import numpy as np
from scipy.signal import savgol_filter
from kneed import KneeLocator
import matplotlib.pyplot as plt


ignore_n =1
y = [0.004265530733518591, 0.004561317018794382, 0.004616992750542099, 0.004738485440380151, 0.0047629926939511645, 0.0047848954901302705, 0.004990836665205965, 0.005008629403975766, 0.005019376678992486, 0.005155821841982263, 0.005194064230578098, 0.00570616773051248, 0.005749163939429152, 0.005756707549760784, 0.006430213975482968, 0.006596917500021701, 0.006670554963452903, 0.007301407265500597, 0.007746016789194875, 0.008209422610986804, 0.009411345102007806, 0.009797600390661465, 0.01037742235729745, 0.023003233424886606, 0.0235068783941114, 0.024283408982503026, 0.026535497648387894, 0.027513911516487972, 0.028923766203393004, 0.029727136969769846, 0.030589914065105292, 0.1729010946383941]

x = np.arange(len(y) - ignore_n)


# print('x: ', len(x))
y = np.array(y)
idx_y = np.argsort(y)[::-1]

y = y[idx_y]

# print('y: ' ,len(y))

y = y[ignore_n:]
# print('new y: ' ,len(y))


# window = 3
# smoothed_y = savgol_filter(y, window, 1)
# print(smoothed_y)

sensitivity = 1


kn = KneeLocator(x, y, curve='convex', direction='decreasing', S=sensitivity, interp_method='polynomial')
print('kn: ', kn.knee)

# kn_smooth = KneeLocator(x, smoothed_y, curve='convex', direction='decreasing', S=sensitivity, interp_method='polynomial')
# print('smoothed kn: ', kn_smooth.knee)


fig, (ax1, ax2) = plt.subplots(2)


# ax2.set_xlabel('k')
# ax2.set_ylabel('PageRank values')
# ax2.plot(x, smoothed_y, 'rx-')
# ax2.vlines(kn_smooth.knee, plt.ylim()[0], plt.ylim()[1], linestyles='dotted')



ax1.set_xlabel('k')
ax1.set_ylabel('PageRank values')
ax1.plot(x, y, 'bx-')
ax1.vlines(kn.knee, np.min(y), np.max(y), linestyles='dashed')




plt.show()
