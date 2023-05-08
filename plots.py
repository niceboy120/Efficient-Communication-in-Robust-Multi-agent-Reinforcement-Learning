import pickle
import matplotlib.pyplot as plt




with open('results_convergence.pickle', 'rb') as f:
    data = pickle.load(f)

plt.plot(data[0], color="blue", marker="o") #Also plot moving average probably..
plt.xlabel("episode", fontsize=14)
plt.ylabel("score", fontsize=14)
plt.show()



# with open('results_edi.pickle', 'rb') as f:
#     data = pickle.load(f)

# fig,ax = plt.subplots()
# ax.plot(data[0], data[1][1:,0], color="red", marker="o")
# ax.set_xlabel("alpha", fontsize=14)
# ax.set_ylabel("score", color="red", fontsize=14)

# ax2=ax.twinx()
# ax2.plot(data[0], data[1][1:,2], color="blue", marker="o")
# ax2.set_ylabel("communications", color="blue", fontsize=14)
# plt.show()