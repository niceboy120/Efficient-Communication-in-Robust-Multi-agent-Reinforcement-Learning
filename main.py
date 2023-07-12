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
            ENV = 'simple_tag' # 1: simple_tag, 2: simple_tag_elisa, 3: simple_tag_mpc, 3: webots

            train_agents_regular = Train(ENV, chkpt_dir='/trained_nets/regular/')
            train_agents_LRRL = Train(ENV, chkpt_dir='/trained_nets/LRRL/')
            train_agents_LRRL2 = Train(ENV, chkpt_dir='/trained_nets/LRRL2/')
            train_agents_LRRL3 = Train(ENV, chkpt_dir='/trained_nets/LRRL3/')
            train_agents_LRRL4 = Train(ENV, chkpt_dir='/trained_nets/LRRL4/')
            # train_agents_regular.testing(load_alt_location='simple_tag')
            # train_agents_LRRL.testing()

            # Training maddpg agents
            history_regular = train_agents_regular.training(load=False, greedy=True, decreasing_eps=True, lexi_mode=False, log=True, load_alt_location='simple_tag')
            history_LRRL, _ = train_agents_LRRL.training(load=False, greedy=True, decreasing_eps=True, lexi_mode=True, log=True, robust_actor_loss=False, load_alt_location='simple_tag', noise_mode=1)
            history_LRRL2, _ = train_agents_LRRL2.training(load=False, greedy=True, decreasing_eps=True, lexi_mode=True, log=True, robust_actor_loss=False, load_alt_location='simple_tag', noise_mode=2)
            history_LRRL3, _ = train_agents_LRRL3.training(load=False, greedy=True, decreasing_eps=True, lexi_mode=True, log=True, robust_actor_loss=True, load_alt_location='simple_tag', noise_mode=1)
            history_LRRL4, _ = train_agents_LRRL4.training(load=False, greedy=True, decreasing_eps=True, lexi_mode=True, log=True, robust_actor_loss=True, load_alt_location='simple_tag', noise_mode=2)



            # with open('results/results_convergence.pickle', 'wb+') as f:
            #     # pickle.dump([history_regular], f)
            #     pickle.dump([history_regular, history_LRRL], f)

            # Testing policies without noise
            test_regular, _ = train_agents_regular.testing(render=False, noisy=False)
            test_LRRL, _ = train_agents_LRRL.testing(render=False, lexi_mode=True, noisy=False)
            test_LRRL2, _ = train_agents_LRRL2.testing(render=False, lexi_mode=True, noisy=False)
            test_LRRL3, _ = train_agents_LRRL3.testing(render=False, lexi_mode=True, noisy=False)
            test_LRRL4, _ = train_agents_LRRL4.testing(render=False, lexi_mode=True, noisy=False)

            # Testing the robustness
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

            with open('results/results_noise_test.pickle', 'wb+') as f:
                pickle.dump([test_regular, test_LRRL, test_LRRL2, test_LRRL3, test_LRRL4, test_regular_noise_a, test_regular_noise_b, test_LRRL_noise_a, test_LRRL_noise_b, test_LRRL_noise2_a, test_LRRL_noise2_b, test_LRRL_noise3_a, test_LRRL_noise3_b, test_LRRL_noise4_a, test_LRRL_noise4_b], f)

            # # # Training gammanet
            # train_agents_regular.testing(edi_mode='train', edi_load=False, render=False, lexi_mode=False)
            # # train_agents_LRRL.testing(edi_mode='train', edi_load=False, render=False, lexi_mode=True)

            # # Dummytesting gammanet
            # _, sequence = train_agents_regular.testing(render=False, N_games=1)
            # print(sequence[1][0], sequence[2][0], np.linalg.norm(sequence[1][0]-sequence[2][0], np.inf))
            # print(train_agents_regular.gammnet.communication(sequence[1][0], sequence[2][0], 0.0001, print_gamma=True))
            # print(train_agents_regular.gammnet.communication(sequence[1][0], sequence[2][0], 0.001, print_gamma=True))
            # print(train_agents_regular.gammnet.communication(sequence[1][0], sequence[2][0], 0.01, print_gamma=True))
            # print(train_agents_regular.gammnet.communication(sequence[1][0], sequence[2][0], 0.1, print_gamma=True))
            # print(train_agents_regular.gammnet.communication(sequence[1][0], sequence[2][0], 1.0, print_gamma=True))
            # print(train_agents_regular.gammnet.communication(sequence[1][0], sequence[2][0], 10.0, print_gamma=True))
            # print("")
            
            # print(train_agents_regular.gammnet.communication(sequence[38][1], sequence[40][1], 0.0001, print_gamma=True))
            # print(train_agents_regular.gammnet.communication(sequence[38][1], sequence[40][1], 0.001, print_gamma=True))
            # print(train_agents_regular.gammnet.communication(sequence[38][1], sequence[40][1], 0.01, print_gamma=True))
            # print(train_agents_regular.gammnet.communication(sequence[38][1], sequence[40][1], 0.1, print_gamma=True))
            # print(train_agents_regular.gammnet.communication(sequence[38][1], sequence[40][1], 1.0, print_gamma=True))
            # print(train_agents_regular.gammnet.communication(sequence[38][1], sequence[40][1], 10.0, print_gamma=True))
            # print("")

            

            # # Testing with EDI disabled
            # history, _ = train_agents_regular.testing(edi_mode='disabled', render=False, lexi_mode=False)
            # mean_regular = np.mean(history, axis=0)
            # std_regular = np.std(history, axis=0)

            # # history, _ = train_agents_LRRL.testing(edi_mode='disabled', render=False, lexi_mode=True)
            # # mean_LRRL = np.mean(history, axis=0)
            # # std_LRRL = np.std(history, axis=0)

            # # Testing EDI for different zetas
            # zeta = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]
            # for z in zeta:
            #     history, _ = train_agents_regular.testing(edi_mode='test', render=False, zeta=z, lexi_mode=False)
            #     mean_regular = np.vstack((mean_regular, np.mean(history, axis=0)))
            #     std_regular = np.vstack((std_regular, np.std(history, axis=0)))

            #     # history, _ = train_agents_LRRL.testing(edi_mode='test', render=False, zeta=z, lexi_mode=True)
            #     # mean_LRRL = np.vstack((mean_LRRL, np.mean(history, axis=0)))
            #     # std_LRRL = np.vstack((std_LRRL, np.std(history, axis=0)))

            # # Dumping output
            # with open('results/results_edi.pickle', 'wb+') as f:
            #     pickle.dump([zeta, mean_regular, std_regular],f)
            #     # pickle.dump([zeta, mean_regular, std_regular, mean_LRRL, std_LRRL],f)

                
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
