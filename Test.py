# import numpy as np


# def bound(x):
#     if x < 0.8:
#         return 0
#     if x < 1.0:
#         return (x - 0.8) * 100
#     return min((20+(x-1)*15), 40)

# x = np.linspace(0,5,1001)
# print(x)

# y = []

# for i in range(len(x)):
#     y.append(bound(x[i]))

# import matplotlib.pyplot as plt

# plt.plot(x,y)
# plt.show()


mask = [1,4,7, 11, 3, 67, 10]

while len(mask)>2:
    mask.remove(max(mask))

print(mask)