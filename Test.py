from train_agent import Train

# train_agents = Train('simple_tag')
# train_agents.testing(edi_load=False)




import time
import numpy as np
import pickle

from MPE.make_env import make_env
env = make_env('simple_tag')
obs = env.reset()



for i in range(10):
    env.render()
    print(obs)

    no_op = np.array([0, 0.1, 0.12, 0.33, 0.54])
    action = [no_op, no_op, no_op, no_op]
    obs, reward, done, info = env.step(action)

    print("")
    print(obs[0])
    print(obs[1])
    print(obs[2])
    print(obs[3])


    mask_agent = [0,1,2]
    mask_msg = [2,3]
    comm = []
    for i in mask_agent:
        comm.append(obs[i][mask_msg])
    print(comm)
    input("Press Enter to continue...")