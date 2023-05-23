import math
import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats


def sigmoid(x, b=1):
    a = []
    for item in x:
        a.append(1 / (1 + math.exp(-b * item)))
    return np.array(a)


mu1 = -2
mu2 = 2
variance = 1.5
sigma = math.sqrt(variance)

x = np.linspace(mu1 - 3 * sigma, mu2 + 3 * sigma, 100)

y1 = stats.norm.pdf(x, mu1, sigma)
y2 = stats.norm.pdf(x, mu2, sigma)

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

ax1.plot(x, 0.7 * y1 + 0.3 * y2)
# ax1.plot(x, 0.7 * y1 + 0.3 * y2, c='grey', alpha=0.3)
# ax1.plot(x, 0.3 * y2)
ax1.plot(x, 0.3 * y2, c='grey', alpha=0.3)
# ax1.plot(x, 0.7 * y1)
ax1.plot(x, 0.7 * y1, c='grey', alpha=0.3)

ys1 = sigmoid(x - mu1, b=-5)
ys2 = sigmoid(x - mu2, b=-5)

y = (0.7 * y1 * ys1 + 0.3 * y2 * ys2) / (0.7 * y1 + 0.3 * y2)
fedy = 0.7 * ys1 + 0.3 * ys2
ax2.plot(x, fedy, c='r')
ax2.plot(x, 0.5 * np.ones_like(x), '--', c='orange')

# plt.scatter(x1, y1, c='c', label='normal (0~8)')
# plt.scatter(x2, y2, c='r', label='permuted label (0~8)')
# plt.scatter(x3, y3, c='orange', label='OOD random label (9)')
# ax1.set_ylim(0, 0.5)
ax1.set_xlabel('$x$')
ax1.set_ylabel('$p(x)$')
ax2.set_ylabel('$p(y|x)$')

# plt.legend()
plt.show()
os.makedirs('visualization', exist_ok=True)
# plt.savefig('visualization/fedem.png')
