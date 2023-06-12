from train_agent import Train
import numpy as np
import pickle

# def training(self, edi_mode='disabled', load=True, load_adversaries=True, edi_load=True, render=False, zeta=0.0, greedy=False, decreasing_eps=True, N_games=None, lexi_mode=False, robust_actor_loss=True, noisy=False)
# def testing(self, edi_mode='disabled', load=True, load_adversaries=True, edi_load=True, render=True, zeta=0.0, greedy=False, decreasing_eps=False, N_games=None, lexi_mode=False, robust_actor_loss=True, noisy=False)

if __name__ == '__main__':
    stop = 0

    while True:
        if stop > 0:
            break

        try:
            train_agents_regular = Train('simple_tag', chkpt_dir='/trained_nets/regular/')
            train_agents_LRRL = Train('simple_tag', chkpt_dir='/trained_nets/LRRL/')
            # train_agents_regular.testing()
            # train_agents_LRRL.testing()

            # Training maddpg agents
            history_regular = train_agents_regular.training(load=True, greedy=True, decreasing_eps=True, lexi_mode=False, log=True)
            history_LRRL = train_agents_LRRL.training(load=True, greedy=True, decreasing_eps=True, lexi_mode=True, log=True, robust_actor_loss=True)

            with open('results/results_convergence.pickle', 'wb+') as f:
                # pickle.dump([history_regular], f)
                pickle.dump([history_regular, history_LRRL], f)

            # Testing the robustness
            test_regular_noise = train_agents_regular.testing(render=False, noisy=True)
            test_LRRL_noise = train_agents_LRRL.testing(render=False, noisy=True)

            with open('results/results_noise_test.pickle', 'wb+') as f:
                pickle.dump([test_regular_noise, test_LRRL_noise], f)

            # Training gammanet
            train_agents_regular.testing(edi_mode='train', edi_load=False, render=False, lexi_mode=False)
            train_agents_LRRL.testing(edi_mode='train', edi_load=False, render=False, lexi_mode=True)

            # Testing with EDI disabled
            history = train_agents_regular.testing(edi_mode='disabled', render=False, lexi_mode=False)
            mean_regular = np.mean(history, axis=0)
            std_regular = np.std(history, axis=0)

            history = train_agents_LRRL.testing(edi_mode='disabled', render=False, lexi_mode=True)
            mean_LRRL = np.mean(history, axis=0)
            std_LRRL = np.std(history, axis=0)

            # Testing EDI for different alphas
            zeta = [1,10,100,500,1000]
            for z in zeta:
                print("Testing with zeta = ", z)
                history = train_agents_regular.testing(edi_mode='test', render=False, zeta=z, lexi_mode=False)
                mean_regular = np.vstack((mean_regular, np.mean(history, axis=0)))
                std_regular = np.vstack((std_regular, np.std(history, axis=0)))

                history = train_agents_LRRL.testing(edi_mode='test', render=False, zeta=z, lexi_mode=True)
                mean_LRRL = np.vstack((mean_LRRL, np.mean(history, axis=0)))
                std_LRRL = np.vstack((std_LRRL, np.std(history, axis=0)))

            # Dumping output
            with open('results/results_edi.pickle', 'wb+') as f:
                # pickle.dump([alpha, mean_regular, std_regular],f)
                pickle.dump([zeta, mean_regular, std_regular, mean_LRRL, std_LRRL],f)


            # Want to make it so it does not always overwrite the picle file. maybe add to it?
            # Change name alpha to zeta?
            

            


            # train_agents.testing()

                
        except KeyboardInterrupt:
            train_agents_regular.ask_save()
            # train_agents_LRRL.ask_save()
            print("Paused, hit ENTER to continue, type q to quit.")
            response = input()
            if response == 'q':
                # with open('interrupted_results.pickle', 'wb+') as f:
                #     pickle.dump(history,f)
                break
            else:
                print('Resuming...')
                continue

        stop += 1 







""" 
To Dos:
- change alpha to zeta
- make a config file/class in utils
- implement a convergence check
    - For reward mode 1 maybe for all adversaries the average distance to the target
    - For reward mode 2 maybe if multiple episodes in a row have had double tags
- Do the thing that the output pickle file does not get overwritten everytime but maybe add to it?

Need to find a way to encourage the second one to catch up without penalizing the first one for going forward

Maybe don't do a full clear of the replay buffer but a first-in-first-out method with a fixed size might be better??? ALREADY BUILT IN, JUST MAKE THE BUFFER SIZE SMALLER!!
"""



                


