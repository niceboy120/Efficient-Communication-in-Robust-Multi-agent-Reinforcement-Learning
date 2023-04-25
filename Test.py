import pickle
import matplotlib.pyplot as plt

with open('results.pickle', 'rb') as f:
    data = pickle.load(f)

print(data)

fig,ax = plt.subplots()
ax.plot(data[0], data[1][1:], color="red", marker="o")
ax.set_xlabel("alpha", fontsize=14)
ax.set_ylabel("score", color="red", fontsize=14)

ax2=ax.twinx()
ax2.plot(data[0], data[3][1:], color="blue", marker="o")
ax2.set_ylabel("communications", color="blue", fontsize=14)
plt.show()