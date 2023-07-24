from train_agent import Train
import numpy as np
import pickle
import random

# def training(self, edi_mode='disabled', load=True, load_adversaries=True, edi_load=True, render=False, zeta=0.0, greedy=False, decreasing_eps=True, N_games=None, lexi_mode=False, robust_actor_loss=True, noisy=False)
# def testing(self, edi_mode='disabled', load=True, load_adversaries=True, edi_load=True, render=True, zeta=0.0, greedy=False, decreasing_eps=False, N_games=None, lexi_mode=False, robust_actor_loss=True, noisy=False)

if __name__ == '__main__':
    stop = 0

    while True:
        if stop > 0:
            break

        try:
            ENV = 'simple_tag' # 1: simple_tag, 2: simple_tag_elisa, 3: simple_tag_mpc, 3: simple_tag_webots

            train_agents_regular = Train(ENV, chkpt_dir='/trained_nets/regular/')
            train_agents_LRRL = Train(ENV, chkpt_dir='/trained_nets/LRRL/')
            train_agents_LRRL2 = Train(ENV, chkpt_dir='/trained_nets/LRRL2/')
            train_agents_LRRL3 = Train(ENV, chkpt_dir='/trained_nets/LRRL3/')
            train_agents_LRRL4 = Train(ENV, chkpt_dir='/trained_nets/LRRL4/')
                
            # history_regular, _ =  train_agents_regular.testing(render=False, noisy=False, load_alt_location='simple_tag')
            # history_LRRL, _ = train_agents_LRRL4.testing(render=False, noisy=False, load_alt_location='simple_tag') 
            # with open('results/'+ENV+'/results_policy_previous_env.pickle', 'wb') as f:
            #     pickle.dump([history_regular, history_LRRL], f)


            # train_agents_regular.testing()
            # train_agents_LRRL.testing()
            # train_agents_LRRL2.testing()
            # train_agents_LRRL3.testing()
            # train_agents_LRRL4.testing()

            # Training maddpg agents
            # history_regular = train_agents_regular.training(load=False, greedy=True, decreasing_eps=True, lexi_mode=False, log=True)#, load_alt_location='simple_tag')
            history_LRRL, _ = train_agents_LRRL.training(load=False, greedy=True, decreasing_eps=True, lexi_mode=True, log=True, robust_actor_loss=False, noise_mode=1)#, load_alt_location='simple_tag')
            history_LRRL2, _ = train_agents_LRRL2.training(load=False, greedy=True, decreasing_eps=True, lexi_mode=True, log=True, robust_actor_loss=False, noise_mode=2)#, load_alt_location='simple_tag')
            history_LRRL3, _ = train_agents_LRRL3.training(load=False, greedy=True, decreasing_eps=True, lexi_mode=True, log=True, robust_actor_loss=True, noise_mode=1)#, load_alt_location='simple_tag')
            history_LRRL4, _ = train_agents_LRRL4.training(load=False, greedy=True, decreasing_eps=True, lexi_mode=True, log=True, robust_actor_loss=True, noise_mode=2)#, load_alt_location='simple_tag')


            # with open('results/'+ENV+'/results_convergence.pickle', 'wb+') as f:
            #     pickle.dump([history_regular], f)
            #     pickle.dump([history_regular, history_LRRL], f)
 
            # # # Testing policies without noise
            test_regular, _ = train_agents_regular.testing(render=False, noisy=False)
            test_LRRL, _ = train_agents_LRRL.testing(render=False, lexi_mode=True, noisy=False)
            test_LRRL2, _ = train_agents_LRRL2.testing(render=False, lexi_mode=True, noisy=False)
            test_LRRL3, _ = train_agents_LRRL3.testing(render=False, lexi_mode=True, noisy=False)
            test_LRRL4, _ = train_agents_LRRL4.testing(render=False, lexi_mode=True, noisy=False)

            # # # Testing the robustness
            test_regular_noise_a, _ = train_agents_regular.testing(render=False, noisy=True, noise_mode=1)
            test_regular_noise_b, _ = train_agents_regular.testing(render=False, noisy=True, noise_mode=2)
            test_LRRL_noise_a, _ = train_agents_LRRL.testing(render=False, lexi_mode=True, noisy=True, noise_mode=1)
            test_LRRL_noise_b, _ = train_agents_LRRL.testing(render=False, lexi_mode=True, noisy=True, noise_mode=2)
            test_LRRL_noise2_a, _ = train_agents_LRRL2.testing(render=False, lexi_mode=True, noisy=True, noise_mode=1)
            test_LRRL_noise2_b, _ = train_agents_LRRL2.testing(render=False, lexi_mode=True, noisy=True, noise_mode=2)
            test_LRRL_noise3_a, _ = train_agents_LRRL3.testing(render=False, lexi_mode=True, noisy=True, noise_mode=1)
            test_LRRL_noise3_b, _ = train_agents_LRRL3.testing(render=False, lexi_mode=True, noisy=True, noise_mode=2)
            test_LRRL_noise4_a, _ = train_agents_LRRL4.testing(render=False, lexi_mode=True, noisy=True, noise_mode=1)
            test_LRRL_noise4_b, _ = train_agents_LRRL4.testing(render=False, lexi_mode=True, noisy=True, noise_mode=2)

            with open('results/'+ENV+'/results_noise_test.pickle', 'wb+') as f:
                pickle.dump([test_regular, test_LRRL, test_LRRL2, test_LRRL3, test_LRRL4, test_regular_noise_a, test_regular_noise_b, test_LRRL_noise_a, test_LRRL_noise_b, test_LRRL_noise2_a, test_LRRL_noise2_b, test_LRRL_noise3_a, test_LRRL_noise3_b, test_LRRL_noise4_a, test_LRRL_noise4_b], f)

            # with open('results/'+ENV+'/results_noise_test_2.pickle', 'wb') as f:
            #     pickle.dump([test_regular, test_LRRL4, test_regular_noise_a, test_regular_noise_b, test_LRRL_noise4_a, test_LRRL_noise4_b], f)

            # # # Training gammanet
            # train_agents_regular.testing(edi_mode='train', edi_load=True, render=False, lexi_mode=False)
            # # # train_agents_LRRL4.testing(edi_mode='train', edi_load=False, render=False, lexi_mode=True)

            # # Dummytesting gammanet
            # for i in range(5):
            #     _, sequence = train_agents_regular.testing(render=False, N_games=1)
            #     print(sequence[1][0], sequence[2][0], np.linalg.norm(sequence[1][0]-sequence[2][0], np.inf))
            #     print(train_agents_regular.gammanet.communication(sequence[1][0], sequence[5][0], 5, return_gamma=True, load_net=True))
            #     print(train_agents_regular.gammanet.communication(sequence[1][0], sequence[10][0], 5, return_gamma=True, load_net=True))
            #     print(train_agents_regular.gammanet.communication(sequence[1][0], sequence[15][0], 5, return_gamma=True, load_net=True))
            #     print(train_agents_regular.gammanet.communication(sequence[1][0], sequence[20][0], 5, return_gamma=True, load_net=True))
            #     print(train_agents_regular.gammanet.communication(sequence[1][0], sequence[25][0], 5, return_gamma=True, load_net=True))
            #     print(train_agents_regular.gammanet.communication(sequence[1][0], sequence[30][0], 5, return_gamma=True, load_net=True))
            #     print(train_agents_regular.gammanet.communication(sequence[1][0], sequence[35][0], 5, return_gamma=True, load_net=True))
            #     print("")
                
            #     print(train_agents_regular.gammanet.communication(sequence[38][1], sequence[40][1], 5, return_gamma=True, load_net=True))
            #     print(train_agents_regular.gammanet.communication(sequence[38][1], sequence[42][1], 5, return_gamma=True, load_net=True))
            #     print(train_agents_regular.gammanet.communication(sequence[38][1], sequence[44][1], 5, return_gamma=True, load_net=True))
            #     print(train_agents_regular.gammanet.communication(sequence[38][1], sequence[46][1], 5, return_gamma=True, load_net=True))
            #     print(train_agents_regular.gammanet.communication(sequence[38][1], sequence[48][1], 5, return_gamma=True, load_net=True))
            #     print(train_agents_regular.gammanet.communication(sequence[38][1], sequence[50][1], 5, return_gamma=True, load_net=True))
            #     print(train_agents_regular.gammanet.communication(sequence[38][1], sequence[52][1], 5, return_gamma=True, load_net=True))
            #     print("")

            # zeta_diff = [[]]
            # for i in range(1000):
            #     _, sequence = train_agents_regular.testing(render=False, N_games=1)
            #     start_state = random.randint(0, 20)
            #     agent_idx = random.randint(0,1)
            #     for j in range(start_state+1, len(sequence)):
            #         comm, zeta = train_agents_regular.gammanet.communication(sequence[start_state][agent_idx], sequence[j][agent_idx], 0, return_gamma=True, load_net=True)
            #         diff = j-start_state
            #         if diff >= len(zeta_diff):
            #             zeta_diff.append([])
            #         zeta_diff[diff].append(zeta)
            
            # with open('results/'+ENV+'/results_zeta_diff.pickle', 'wb+') as f:
            #     pickle.dump(zeta_diff, f)            

            # # Testing with EDI disabled
            # history, _ = train_agents_regular.testing(edi_mode='disabled', render=False, lexi_mode=False)
            # mean_regular = np.mean(history, axis=0)
            # std_regular = np.std(history, axis=0)
            # worst_regular = np.min(history, axis=0)

            # history, _ = train_agents_LRRL4.testing(edi_mode='disabled', render=False, lexi_mode=True)
            # mean_LRRL = np.mean(history, axis=0)
            # std_LRRL = np.std(history, axis=0)
            # worst_LRRL = np.min(history, axis=0)

            # # Testing EDI for different zetas
            # zeta = [1, 1.5, 2, 2.5, 3, 3.1, 3.2, 3.3, 3.4, 3.5, 4]
            # for z in zeta:
            #     history, _ = train_agents_regular.testing(edi_mode='test', render=False, zeta=z, lexi_mode=False)
            #     mean_regular = np.vstack((mean_regular, np.mean(history, axis=0)))
            #     std_regular = np.vstack((std_regular, np.std(history, axis=0)))
            #     worst_regular = np.vstack((worst_regular, np.min(history, axis=0)))

            #     history, _ = train_agents_LRRL4.testing(edi_mode='test', render=False, zeta=z, lexi_mode=True)
            #     mean_LRRL = np.vstack((mean_LRRL, np.mean(history, axis=0)))
            #     std_LRRL = np.vstack((std_LRRL, np.std(history, axis=0)))
            #     worst_LRRL = np.vstack((worst_LRRL, np.min(history, axis=0)))

            # # Dumping output
            # with open('results/'+ENV+'/results_edi.pickle', 'wb+') as f:
            #     # pickle.dump([zeta, mean_regular, std_regular, worst_regular],f)
            #     pickle.dump([zeta, mean_regular, std_regular, worst_regular, mean_LRRL, std_LRRL, worst_LRRL],f)

                
        except KeyboardInterrupt:
            train_agents_regular.ask_save()
            train_agents_LRRL4.ask_save()
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
