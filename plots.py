import pickle
import matplotlib.pyplot as plt
import numpy as np


ENV = 'simple_tag'



"""
=========================================================================================================================
            CONVERGENCE
=========================================================================================================================
"""


# with open('results/'+ENV+'/results_convergence.pickle', 'rb') as f:
#     data = pickle.load(f)

# # print(data[0])

# window_size = 300
# mavg_adv = []
# mavg_ag = []

# for i in range(len(data[0])):
#     if i<=(window_size+1):
#         mavg_adv.append(np.mean(data[0][0:i]))
#         mavg_ag.append(np.mean(data[1][0:i]))
#     else:
#         mavg_adv.append(np.mean(data[0][i-window_size:i]))
#         mavg_ag.append(np.mean(data[1][i-window_size:i]))

# fig, axs = plt.subplots(2)
# fig.suptitle("Convergence of MADDPG algorithm over "+str(len(data))+" episodes")
# axs[0].plot(data[0], color="blue", label="Score per episode")
# axs[0].plot(mavg_adv, color="red", marker=".", label="Moving average of "+str(window_size))
# axs[0].set(ylabel="score adversaries")
# # axs[0].legend()

# axs[1].plot(data[1], color="blue", label="Score per episode")
# axs[1].plot(mavg_ag, color="red", marker=".", label="Moving average of "+str(window_size))
# axs[1].set(xlabel="episode", ylabel="score agents")
# axs[1].legend()
# plt.show()


"""
=========================================================================================================================
            ZETA FOR STATE STEPS
=========================================================================================================================
"""


with open('results/'+ENV+'/results_zeta_diff.pickle', 'rb') as f:
    data = pickle.load(f)

for i in range(1, len(data)):
    data[i] = np.mean(data[i])

plt.plot(data[1:])
plt.xlabel('Steps between states')
plt.ylabel('$\zeta$ from neural network')
plt.title('Increase of $\zeta$ for states further apart')
plt.show()


"""
=========================================================================================================================
            PERFORMANCE COMM PLOT
=========================================================================================================================
"""

with open('results/'+ENV+'/results_edi.pickle', 'rb') as f:
    data = pickle.load(f)

# [zeta, mean_regular, std_regular, worst_regular, mean_LRRL, std_LRRL, worst_LRRL]
# For all but zeta: 1st column is score adveraries, 2nd is score agents, 3rd is communications. and 1st row is without EDI

from utils import HyperParameters
par = HyperParameters()

for i in range(len(data[0])):
    print("zeta th: %.3f, limit: %.1f, avg: %.1f, worst: %.1f" % (data[0][i], data[0][i]*(1/(1-par.gamma)), data[1][0,0]-data[1][i+1, 0], data[1][0,0]-data[3][i+1, 0]))


fig,ax = plt.subplots()
ax.fill_between(data[0], data[1][1:,0]+data[2][1:,0], data[1][1:,0]-data[2][1:,0], color="red", alpha=0.3)
ax.fill_between(data[0], data[4][1:,0]+data[5][1:,0], data[4][1:,0]-data[5][1:,0], color="tomato", alpha=0.3)
ax.plot(data[0], data[1][1:,0], color="red", marker="o", label='Vanilla')
ax.plot(data[0], data[4][1:,0], color="tomato", marker=".", linestyle="--", label='LRRL')
ax.set_xlabel("$\zeta_{\mathrm{th}}$", fontsize=14)
ax.set_ylabel("score adversaries", color="red", fontsize=14)
plt.legend()

ax2=ax.twinx()
ax2.fill_between(data[0], data[1][1:,2]+data[2][1:,2], data[1][1:,2]-data[2][1:,2], color="blue", alpha=0.3)
ax2.fill_between(data[0], data[4][1:,2]+data[5][1:,2], data[4][1:,2]-data[5][1:,2], color="cornflowerblue", alpha=0.3)
ax2.plot(data[0], data[1][1:,2], color="blue", marker="o")
ax2.plot(data[0], data[4][1:,2], color="cornflowerblue", marker=".", linestyle="--")
ax2.set_ylabel("communications", color="blue", fontsize=14)
plt.title("Number of communications and score for different $\zeta_{\mathrm{th}}$ values")
plt.show()



"""
=========================================================================================================================
            NOISE TEST
=========================================================================================================================
"""

with open('results/'+ENV+'/results_noise_test.pickle', 'rb') as f:
    data = pickle.load(f)
dic = ["Vanilla no noise  ", "LRRL1(T1) no noise", "LRRL1(T2) no noise", "LRRL2(T1) no noise", "LRRL2(T2) no noise", "Vanilla noise 1   ", "Vanilla noise 2   ", "LRRL1(T1) noise 1 ", "LRRL1(T1) noise 2 ", "LRRL1(T2) noise 1 ", "LRRL1(T2) noise 2 ", "LRRL2(T1) noise 1 ", "LRRL2(T1) noise 2 ", "LRRL2(T2) noise 1 ", "LRRL2(T2) noise 2 "]

# with open('results/'+ENV+'/results_noise_test_2.pickle', 'rb') as f:
#     data = pickle.load(f)
# dic = ["Vanilla no noise  ", "LRRL2(T2) no noise", "Vanilla noise 1   ", "Vanilla noise 2   ", "LRRL2(T2) noise 1 ", "LRRL2(T2) noise 2 "]


mean = []
std = []
for i in range(len(data)):
    score = []
    for j in range(len(data[i])):
        score.append(data[i][j][0])
    mean.append(np.mean(score))
    std.append(np.std(score))
    print(dic[i], ", mean: %.1f, std: %.1f" % (mean[i], std[i]))

