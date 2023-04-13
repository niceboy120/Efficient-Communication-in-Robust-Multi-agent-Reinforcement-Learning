from MADDPG.train_agent import Train
import time


""" 
General script parameters
"""
TRAIN_OR_TEST = True


"""
Parameters MADDPG agents:
"""
MODE = 'train' #pick "train" or "test"
LOAD = True
SAVE = True
PRINT_INTERVAL = 500
# N_GAMES = 10000
# MAX_STEPS = 50



"""
Parameters robustness surrogates
"""
TRAIN_SVR = False



"""
Parameters lexicographic robust reinforcement learning
"""
LRRL = False




if __name__ == '__main__':
    if TRAIN_OR_TEST:
        train_agents = Train(MODE, LOAD, SAVE)

        try:
            train_agents.loop()
        except KeyboardInterrupt:
            train_agents.ask_save()


                











                


