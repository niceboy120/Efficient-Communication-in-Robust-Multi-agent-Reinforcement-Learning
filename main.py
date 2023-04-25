from train_agent import Train
import time
import numpy as np
import pickle



""" 
General script parameters
"""
TRAIN_OR_TEST = True


"""
Parameters MADDPG agents:
"""
MADDPG_MODE = 'test' #pick "train" or "test"
LOAD = True
SAVE = False
PRINT_INTERVAL = 500



"""
Parameters robustness surrogates
"""
EDI_MODE = 'disabled' #Pick "parallel", "sequential" , "test" or "disabled"
EDI_LOAD = False
EDI_SAVE = True

"""
Parameters lexicographic robust reinforcement learning
"""
LRRL = False


# alpha = [0.4, 0.6, 0.8]
alpha = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


if __name__ == '__main__':
    if TRAIN_OR_TEST:
        # for a in alpha:
        #     try:
        #         train_agents = Train('train', 'parallel', load=True, save=True, edi_load=True, edi_save=True, alpha=a, N_games=1000, print_interval=100)
        #         train_agents.loop()
        #     except KeyboardInterrupt:
        #         train_agents.ask_save()
        
        # score_mean = []
        # score_std = []
        # communication_mean = []
        # communication_std = []

        # train_agents = Train('test', 'disabled', load=True, save=False, edi_load=False, edi_save=False, N_games=1000, render=False)
        # score_history, communication_history = train_agents.loop()

        # score_mean.append(np.mean(score_history))
        # score_std.append(np.std(score_history))
        # communication_mean.append(np.mean(communication_history))
        # communication_std.append(np.std(communication_history))        

        # for a in alpha:
        #     train_agents = Train('test', 'test', load=True, save=False, edi_load=True, edi_save=False, alpha=a, N_games=1000, render=False)
        #     score_history, communication_history = train_agents.loop()
        #     score_mean.append(np.mean(score_history))
        #     score_std.append(np.std(score_history))
        #     communication_mean.append(np.mean(communication_history))
        #     communication_std.append(np.std(communication_history))
        # with open('results.pickle', 'wb+') as f:
        #     pickle.dump([alpha, score_mean, score_std, communication_mean, communication_std],f)


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
                











                


