import os

import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data

dic = torch.load('ood.pt')

# np.random.seed(1)

x1 = dic['test:x']
y1 = dic['test:y']
idx = np.random.choice(len(x1), 900)
x1 = x1[idx]
y1 = y1[idx]

x2 = dic['testr:x']
y2 = dic['testr:y']
idx = np.random.choice(len(x2), 2000)
x2 = x2[idx]
y2 = y2[idx]

x3 = dic['ood:x']
y3 = dic['ood:y']
idx = np.random.choice(len(x3), 100)
x3 = x3[idx]
y3 = y3[idx]

plt.scatter(x1, y1, c='c', label='normal (excluding 1)', s=10)
# plt.scatter(x2, y2, c='r', label='permuted label (excluding 1)', s=0.1)
plt.scatter(x3, y3, c='orange', label='OOD random label (1)', s=10)

plt.xlabel('$\log p(x)$')
plt.ylabel('$\log p(y|x)$')

# seaborn.scatterplot(data=data_dict, x='x1', y='y1')


# seaborn.scatterplot(data=data_dict, x='x2', y='y2')
plt.legend()
plt.show()
os.makedirs('visualization', exist_ok=True)
# plt.savefig('visualization/ood.png')
