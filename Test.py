# from train_agent import Train

# train_agents = Train('simple_tag')
# train_agents.testing(edi_mode='test', edi_load=False)


import pickle
import matplotlib.pyplot as plt

with open('results.pickle', 'rb') as f:
        data = pickle.load(f)

print(data)

fig,ax = plt.subplots()
ax.plot([-0.2]+data[0], data[1][:,0], color="red", marker="o")
ax.set_xlabel("alpha", fontsize=14)
ax.set_ylabel("score", color="red", fontsize=14)

ax2=ax.twinx()
ax2.plot([-0.2]+data[0], data[1][:,2], color="blue", marker="o")
ax2.set_ylabel("communications", color="blue", fontsize=14)
plt.show()





# import time
# import numpy as np
# import pickle

# from MPE.make_env import make_env
# env = make_env('simple_tag')
# obs = env.reset()



# for i in range(10):
#     env.render()
#     print(obs)

#     no_op = np.array([0, 0.1, 0.12, 0.33, 0.54])
#     action = [no_op, no_op, no_op, no_op]
#     obs, reward, done, info = env.step(action)

#     print("")
#     print(obs[0])
#     print(obs[1])
#     print(obs[2])
#     print(obs[3])


#     mask_agent = [0,1,2]
#     mask_msg = [2,3]

#     comm = []
#     knowledge = []

#     for i in mask_agent:
#         comm.append(obs[i][mask_msg])
#         knowledge.append(obs[i])
    
#     print(comm)

    
    
#     print(knowledge)

    
#     input("Press Enter to continue...")