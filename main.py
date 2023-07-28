from train_agent import Train
import numpy as np
import pickle
import random


if __name__ == '__main__':
    stop = 0

    while True:
        if stop > 0:
            break

        try:
            ENV = 'simple_tag_mpc' # 1: simple_tag, 2: simple_tag_mpc, 3: simple_tag_webots
            if ENV=='simple_tag':
                load_alt_location = None
            else:
                load_alt_location = 'simple_tag'

            # Defining Train instances
            train_agents_regular = Train(ENV, chkpt_dir='/trained_nets/regular/')
            train_agents_LRRL = Train(ENV, chkpt_dir='/trained_nets/LRRL4/')
            train_agents_LRRL_alt = Train(ENV, chkpt_dir='/trained_nets/LRRL3/')
            
            # Testing with the policies from earlier environments
            if ENV=='simple_tag_mpc':
                history_regular, _ =  train_agents_regular.testing(render=False, noisy=False, load_alt_location=load_alt_location)
                history_LRRL, _ = train_agents_LRRL.testing(render=False, noisy=False, load_alt_location=load_alt_location) 
                with open('results/'+ENV+'/results_policy_previous_env.pickle', 'wb') as f:
                    pickle.dump([history_regular, history_LRRL], f)

            # # Rendering policies
            # train_agents_regular.testing()
            # train_agents_LRRL.testing()

            # # Training maddpg agents
            # history_regular = train_agents_regular.training(load=False, greedy=True, decreasing_eps=True, lexi_mode=False, log=True, load_alt_location=load_alt_location)
            # history_LRRL, _ = train_agents_LRRL.training(load=False, greedy=True, decreasing_eps=True, lexi_mode=True, log=True, robust_actor_loss=False, noise_mode=1, load_alt_location=load_alt_location)

 
            # Testing policies without noise
            test_regular, _ = train_agents_regular.testing(render=False, noisy=False)
            test_LRRL, _ = train_agents_LRRL.testing(render=False, lexi_mode=True, noisy=False)

            if ENV=='simple_tag':
                test_LRRL_alt, _ = train_agents_LRRL_alt.testing(render=False, lexi_mode=True, noisy=False)

            # Testing the robustness
            test_regular_noise_a, _ = train_agents_regular.testing(render=False, noisy=True, noise_mode=1)
            test_regular_noise_b, _ = train_agents_regular.testing(render=False, noisy=True, noise_mode=2)
            test_LRRL_noise_a, _ = train_agents_LRRL.testing(render=False, lexi_mode=True, noisy=True, noise_mode=1)
            test_LRRL_noise_b, _ = train_agents_LRRL.testing(render=False, lexi_mode=True, noisy=True, noise_mode=2)
            
            if ENV=='simple_tag':
                test_LRRL_alt_noise_a, _ = train_agents_LRRL_alt.testing(render=False, lexi_mode=True, noisy=True, noise_mode=1)
                test_LRRL_alt_noise_b, _ = train_agents_LRRL_alt.testing(render=False, lexi_mode=True, noisy=True, noise_mode=2)

            with open('results/'+ENV+'/results_noise_test.pickle', 'wb+') as f:
                if ENV=='simple_tag':
                    pickle.dump([test_regular, test_LRRL, test_LRRL_alt, test_regular_noise_a, test_LRRL_noise_a, test_LRRL_alt_noise_a, test_regular_noise_b, test_LRRL_noise_b, test_LRRL_alt_noise_b], f)
                else:
                    pickle.dump([test_regular, test_LRRL, test_regular_noise_a, test_LRRL_noise_a, test_regular_noise_b, test_LRRL_noise_b], f)


            # # Training gammanet
            # train_agents_regular.testing(edi_mode='train', edi_load=False, render=False, lexi_mode=False)
            # train_agents_LRRL.testing(edi_mode='train', edi_load=False, render=False, lexi_mode=True)

            # Testing zetas for different steps between states
            zeta_diff = [[]]
            zeta_diff_LRRL = [[]]
            for i in range(1000):
                _, sequence = train_agents_regular.testing(render=False, N_games=1)
                _, sequence_LRRL = train_agents_LRRL.testing(render=False, N_games=1)
                start_state = random.randint(0, 20)
                agent_idx = random.randint(0,1)
                for j in range(start_state+1, len(sequence)):
                    comm, zeta = train_agents_regular.gammanet.communication(sequence[start_state][agent_idx], sequence[j][agent_idx], 0, return_gamma=True, load_net=True)
                    comm_LRRL, zeta_LRRL = train_agents_LRRL.gammanet.communication(sequence_LRRL[start_state][agent_idx], sequence[j][agent_idx], 0, return_gamma=True, load_net=True)
                    diff = j-start_state
                    if diff >= len(zeta_diff):
                        zeta_diff.append([])
                        zeta_diff_LRRL.append([])
                    zeta_diff[diff].append(zeta)
                    zeta_diff_LRRL[diff].append(zeta_LRRL)
            
            with open('results/'+ENV+'/results_zeta_diff.pickle', 'wb+') as f:
                pickle.dump([zeta_diff, zeta_diff_LRRL], f)            

            # Testing with EDI disabled
            history, _ = train_agents_regular.testing(edi_mode='disabled', render=False, lexi_mode=False)
            mean_regular = np.mean(history, axis=0)
            std_regular = np.std(history, axis=0)
            worst_regular = np.min(history, axis=0)

            history, _ = train_agents_LRRL.testing(edi_mode='disabled', render=False, lexi_mode=True)
            mean_LRRL = np.mean(history, axis=0)
            std_LRRL = np.std(history, axis=0)
            worst_LRRL = np.min(history, axis=0)

            # Testing EDI for different zetas
            zeta = [0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1, 1.5, 2, 2.5, 3, 3.1, 3.2, 3.3, 3.4, 3.5, 4]
            for z in zeta:
                history, _ = train_agents_regular.testing(edi_mode='test', render=False, zeta=z, lexi_mode=False)
                mean_regular = np.vstack((mean_regular, np.mean(history, axis=0)))
                std_regular = np.vstack((std_regular, np.std(history, axis=0)))
                worst_regular = np.vstack((worst_regular, np.min(history, axis=0)))

                history, _ = train_agents_LRRL.testing(edi_mode='test', render=False, zeta=z, lexi_mode=True)
                mean_LRRL = np.vstack((mean_LRRL, np.mean(history, axis=0)))
                std_LRRL = np.vstack((std_LRRL, np.std(history, axis=0)))
                worst_LRRL = np.vstack((worst_LRRL, np.min(history, axis=0)))

            # Dumping output
            with open('results/'+ENV+'/results_edi.pickle', 'wb+') as f:
                pickle.dump([zeta, mean_regular, std_regular, worst_regular, mean_LRRL, std_LRRL, worst_LRRL],f)
                
        except KeyboardInterrupt:
            train_agents_regular.ask_save()
            train_agents_LRRL.ask_save()
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
