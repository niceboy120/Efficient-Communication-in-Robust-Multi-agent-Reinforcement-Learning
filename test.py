# import numpy as np
# from MPE.make_env import make_env
# import time

# env = make_env('simple_tag_elisa')
# obs = env.reset()

# for i in range(10):
#     env.render()
#     print(obs[0])
#     no_op = np.random.rand(5)
#     action = [no_op, no_op, no_op]
#     obs, reward, done, info, n = env.step(action)
#     input("enter")

# zeta_diff = [[]]

# for i in range(10):
#     for j in range(i+1, 20):
#         diff = j-i

#         if diff >= len(zeta_diff):
#             zeta_diff.append([])

#         zeta_diff[diff].append(j*i)
#         print(diff, zeta_diff[diff])
#         print("")
    
# print(zeta_diff)


# import pickle

# test = 'simple_tag'

# with open('results/'+ENV+'/temp.pickle', 'wb') as f:
#     pickle.dump(4, f)

# with open('results/'+test+'/temp.pickle', 'rb') as f:
#     asdf = pickle.load(f)

# print(asdf)

import pickle
with open('results/simple_tag/results_edi.pickle', 'rb') as f:
    data = pickle.load(f)
    
data2 = [data[0], data[1], data[2], data[3], data[7], data[8], data[9]]

with open('results/simple_tag/results_edi.pickle', 'wb') as f:
    pickle.dump(data2,f)