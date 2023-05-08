from train_agent import Train
import numpy as np
import pickle
import matplotlib.pyplot as plt


# def training(self, edi_mode='disabled', load=True, load_adversaries=True, edi_load=True, render=False, alpha=0.0, decreasing_eps=True, N_games=None):
# def testing(self, edi_mode='disabled', load=True, load_adversaries=True, edi_load=True, render=True, alpha=0.0, decreasing_eps=False, N_games=Nones):


alpha = [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]


if __name__ == '__main__':
    stop = 0

    while True:
        if stop > 0:
            break

        try:
            train_agents = Train('simple_tag')

            # Training maddpg agents
            history = []
            for i in range(2,10):
                if i==0:
                    history_session = train_agents.training(load=False)
                elif i==1:
                    history_session = train_agents.training(load=True, load_adversaries=False)
                else: 
                    history_session = train_agents.training()

                for j in range(len(history_session)):
                    history.append(history_session[j][0])

                # train_agents.testing(N_games = 5)
                    
            with open('results_convergence.pickle', 'wb+') as f:
                pickle.dump([history], f)

            # Testing agents
            # train_agents.testing()


            # Training gammanets for different alphas
            for a in alpha:
                print("Training with alpha = ", a)
                train_agents.training(edi_mode='train', edi_load=False, alpha=a)

            # Testing with EDI disabled
            history = train_agents.testing(edi_mode='disabled', render=False)
            mean = np.mean(history, axis=0)
            std = np.std(history, axis=0)

            # Testing EDI for different alphas
            for a in alpha:
                print("Testing with alpha = ", a)
                history = train_agents.testing(edi_mode='test', render=False, alpha=a)
                mean = np.vstack((mean, np.mean(history, axis=0)))
                std = np.vstack((std, np.std(history, axis=0)))

            # Dumping output
            with open('results_edi.pickle', 'wb+') as f:
                pickle.dump([alpha, mean, std],f)


            # Want to make it so it does not always overwrite the picle file. maybe add to it?
            # Change name alpha to zeta?
            

            


            # train_agents.testing()

                
        except KeyboardInterrupt:
            train_agents.ask_save()
            print("Paused, hit ENTER to continue, type q to quit.")
            response = input()
            if response == 'q':
                break
            else:
                print('Resuming...')
                continue

        stop += 1 











                


