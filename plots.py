import pickle
import matplotlib.pyplot as plt
import numpy as np


ENV = 'simple_tag'


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


# with open('results/'+ENV+'/results_zeta_diff.pickle', 'rb') as f:
#     data = pickle.load(f)

# for i in range(1, len(data)):
#     data[i] = np.mean(data[i])

# print(data)




# with open('results/'+ENV+'/results_edi.pickle', 'rb') as f:
#     data = pickle.load(f)

# print(data)

# fig,ax = plt.subplots()
# ax.fill_between(data[0], data[1][1:,0]+data[2][1:,0], data[1][1:,0]-data[2][1:,0], color="red", alpha=0.3)
# # ax.fill_between(data[0], data[3][1:,0]+data[4][1:,0], data[3][1:,0]-data[4][1:,0], color="tomato", alpha=0.3)
# ax.plot(data[0], data[1][1:,0], color="red", marker="o", label='Vanilla')
# # ax.plot(data[0], data[3][1:,0], color="tomato", marker=".", linestyle="--", label='LRRL')
# ax.set_xlabel("$\zeta_{\mathrm{th}}$", fontsize=14)
# ax.set_ylabel("score adversaries", color="red", fontsize=14)
# plt.legend()

# ax2=ax.twinx()
# ax2.fill_between(data[0], data[1][1:,2]+data[2][1:,2], data[1][1:,2]-data[2][1:,2], color="blue", alpha=0.3)
# # ax2.fill_between(data[0], data[3][1:,2]+data[4][1:,2], data[3][1:,2]-data[4][1:,2], color="cornflowerblue", alpha=0.3)
# ax2.plot(data[0], data[1][1:,2], color="blue", marker="o")
# # ax2.plot(data[0], data[3][1:,2], color="cornflowerblue", marker=".", linestyle="--")
# ax2.set_ylabel("communications", color="blue", fontsize=14)
# plt.title("Number of communications and score for different $\zeta_{\mathrm{th}}$ values")
# plt.show()


with open('results/'+ENV+'/results_noise_test.pickle', 'rb') as f:
    data = pickle.load(f)


mean = []
std = []
dic = ["Vanilla no noise  ", "LRRL1(T1) no noise", "LRRL1(T2) no noise", "LRRL2(T1) no noise", "LRRL2(T2) no noise", "Vanilla noise 1   ", "Vanilla noise 2   ", "LRRL1(T1) noise 1 ", "LRRL1(T1) noise 2 ", "LRRL1(T2) noise 1 ", "LRRL1(T2) noise 2 ", "LRRL2(T1) noise 1 ", "LRRL2(T1) noise 2 ", "LRRL2(T2) noise 1 ", "LRRL2(T2) noise 2 "]
for i in range(len(data)):
    score = []
    for j in range(len(data[i])):
        score.append(data[i][j][0])
    mean.append(np.mean(score))
    std.append(np.std(score))
    print(dic[i], ", mean: %.3f, std: %.3f" % (mean[i], std[i]))

