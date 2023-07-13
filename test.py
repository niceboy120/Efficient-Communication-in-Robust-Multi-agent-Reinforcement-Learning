import numpy as np
from MPE.make_env import make_env
import time

env = make_env('simple_tag_elisa')
obs = env.reset()

for i in range(10):
    env.render()
    print(obs[0])
    no_op = np.random.rand(5)
    action = [no_op, no_op, no_op]
    obs, reward, done, info, n = env.step(action)
    input("enter")

    