import numpy as np


class test:
    def __init__(self):
        self.t = 0

    def increase_waypoint(self):
        self.t += 0.01
        x1 = 0.1*self.t*np.cos(self.t)
        y1 = 0.1*self.t*np.sin(self.t)
        x2 = 0.1*self.t*np.cos(self.t+np.pi/2)
        y2 = 0.1*self.t*np.sin(self.t+np.pi/2)
        x3 = 0.1*self.t*np.cos(self.t+np.pi)
        y3 = 0.1*self.t*np.sin(self.t+np.pi)
        return [x1, y1], [x2, y2], [x3, y3]


# test = test()

# x = []
# y = []

# for i in range(100):
#     test.increase_waypoint()
#     x.append(test.x)
#     y.append(test.y)

# plt.plot(x,y)
# plt.show()