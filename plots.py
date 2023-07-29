import pickle
import matplotlib.pyplot as plt
import numpy as np
# from scipy.signal import savgol_filter
plt.rcParams['axes.facecolor'] = 'lightgrey'


# ENV = 'simple_tag'
# ENV = 'simple_tag_mpc'
ENV = 'simple_tag_webots'


"""
=========================================================================================================================
            PREVIOUS ENV
=========================================================================================================================
"""

if ENV=='simple_tag_mpc':
    with open('results/'+ENV+'/results_policy_previous_env.pickle', 'rb') as f:
        data = pickle.load(f)


    for i in range(len(data)):
        print(np.mean(data[i], axis=0))

if ENV=='simple_tag_webots':
    with open('results/'+ENV+'/results_policy_previous_env.pickle', 'rb') as f:
        data = pickle.load(f)

    for i in range(len(data)):
        print(np.mean(data[i], axis=0))    



"""
=========================================================================================================================
            NOISE TEST
=========================================================================================================================
"""

if ENV!='simple_tag_webots':

    with open('results/'+ENV+'/results_noise_test.pickle', 'rb') as f:
        data = pickle.load(f)

    if ENV=='simple_tag':
        dic = ["Vanilla no noise", "LRRL(T2) no noise", "LRRL(T1) no noise", "Vanilla noise 1", "LRRL(T2) noise 1", "LRRL(T1) noise 1", "Vanilla noise 2", "LRRL(T2) noise 2", "LRRL(T1) nosie 2"]
    else:
        dic = ["Vanilla no noise", "LRRL(T2) no noise", "Vanilla noise 1", "LRRL(T2) noise 1", "Vanilla noise 2", "LRRL(T2) noise 2"]

    mean = []
    std = []
    for i in range(len(data)):
        score = []
        for j in range(len(data[i])):
            score.append(data[i][j][0])
        mean.append(np.mean(score))
        std.append(np.std(score))
        print(dic[i], ", mean: %.1f, std: %.1f" % (mean[i], std[i]))

    if ENV=='simple_tag':
        print("$%.1f \pm %.1f$ & $%.1f \pm %.1f$ & $%.1f \pm %.1f$" % (mean[0], std[0], mean[2], std[2], mean[1], std[1]))
        print("$%.1f \pm %.1f$ & $%.1f \pm %.1f$ & $%.1f \pm %.1f$" % (mean[3], std[3], mean[5], std[5], mean[4], std[4]))
        print("$%.1f \pm %.1f$ & $%.1f \pm %.1f$ & $%.1f \pm %.1f$" % (mean[6], std[6], mean[8], std[8], mean[7], std[7]))
    else:
        print("$%.1f \pm %.1f$ & $%.1f \pm %.1f$" % (mean[0], std[0], mean[1], std[1]))
        print("$%.1f \pm %.1f$ & $%.1f \pm %.1f$" % (mean[2], std[2], mean[3], std[3]))
        print("$%.1f \pm %.1f$ & $%.1f \pm %.1f$" % (mean[4], std[4], mean[5], std[5]))


"""
=========================================================================================================================
            ZETA FOR STATE STEPS
=========================================================================================================================
"""


with open('results/'+ENV+'/results_zeta_diff.pickle', 'rb') as f:
    data = pickle.load(f)


data_reg = []
data_LRRL = []
for i in range(1, len(data[0])):
    data_reg.append(np.mean(data[0][i]))
    data_LRRL.append(np.mean(data[1][i]))

# y_new = savgol_filter(data_reg, 42,4)


plt.plot(data_reg, label="Vanilla")
# plt.plot(y_new, label="Smoothed")
plt.plot(data_LRRL, label="LRRL")
plt.xlabel('Steps between states', fontsize=14)
plt.ylabel('$\zeta$ from neural network', fontsize=14)
plt.title('Increase of $\zeta$ for states further apart', fontsize=14)
plt.legend()
plt.grid()
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

print(len(data))

from utils import HyperParameters
par = HyperParameters()

for i in range(len(data[0])):
    # print("zeta th: %.3f, limit: %.1f, avg: %.1f, worst: %.1f" % (data[0][i], data[0][i]*(1/(1-par.gamma)), data[1][0,0]-data[1][i+1, 0], data[1][0,0]-data[3][i+1, 0]))
    print("$%.1f$ & $%.1f$ & $%.1f$ & $%.1f$ \\" %(data[0][i], data[0][i]*(1/(1-par.gamma)), data[1][0,0]-data[1][i+1, 0], data[1][0,0]-data[3][i+1, 0]))

fig,ax = plt.subplots()
ax.fill_between(data[0], data[1][1:,0]+data[2][1:,0], data[1][1:,0]-data[2][1:,0], color="red", alpha=0.3)
ax.fill_between(data[0], data[4][1:,0]+data[5][1:,0], data[4][1:,0]-data[5][1:,0], color="maroon", alpha=0.3)
ax.plot(data[0], data[1][1:,0], color="red", marker="o", label='Vanilla')
ax.plot(data[0], data[4][1:,0], color="maroon", marker=".", linestyle="--", label='LRRL')
ax.set_xlabel("$\zeta_{\mathrm{th}}$", fontsize=14)
ax.set_ylabel("Return adversaries", color="red", fontsize=14)
plt.legend()

ax2=ax.twinx()
ax2.fill_between(data[0], data[1][1:,2]+data[2][1:,2], data[1][1:,2]-data[2][1:,2], color="blue", alpha=0.3)
ax2.fill_between(data[0], data[4][1:,2]+data[5][1:,2], data[4][1:,2]-data[5][1:,2], color="cornflowerblue", alpha=0.3)
ax2.plot(data[0], data[1][1:,2], color="blue", marker="o")
ax2.plot(data[0], data[4][1:,2], color="cornflowerblue", marker=".", linestyle="--")
ax2.set_ylabel("Number of communications", color="blue", fontsize=14)
plt.title("Number of communications and return for different $\zeta_{\mathrm{th}}$ values", fontsize=14)
plt.show()


plt.plot(data[1][1:,2], data[1][1:,0], color="red", label="Vanilla")
plt.plot(data[4][1:,2], data[4][1:,0], color="blue", label="LRRL")
plt.xlabel("Number of communications", fontsize=14)
plt.ylabel("Return adversaries", fontsize=14)
plt.title("Return versus communications", fontsize=14)
plt.legend()
plt.grid()
plt.show()




"""
=========================================================================================================================
            FINAL COMPARISON
=========================================================================================================================
"""

with open('results/simple_tag/results_edi.pickle', 'rb') as f:
    data_sa = pickle.load(f)

with open('results/simple_tag_mpc/results_edi.pickle', 'rb') as f:
    data_mpc = pickle.load(f)

with open('results/simple_tag_webots/results_edi.pickle', 'rb') as f:
    data_webots = pickle.load(f)

plt.plot(data_sa[4][1:,2], data_sa[4][1:,0]-np.interp(data_sa[4][1:,2], data_sa[1][1:,2], data_sa[1][1:,0]), color="red", label="Env A")
plt.plot(data_mpc[4][1:,2], data_mpc[4][1:,0]-np.interp(data_mpc[4][1:,2], data_mpc[1][1:,2], data_mpc[1][1:,0]), color="blue", label="Env B")
plt.plot(data_webots[4][1:,2], data_webots[4][1:,0]-np.interp(data_webots[4][1:,2], data_webots[1][1:,2], data_webots[1][1:,0]), color="limegreen", label="Env C")
plt.xlabel("Number of communications", fontsize=14)
plt.ylabel("Increase adversary return for LRRL", fontsize=14)
plt.title("Increase adversary return", fontsize=14)
plt.xlim([6,45])
plt.legend()
plt.grid()
plt.show()