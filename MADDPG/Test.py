# from maddpg import MADDPG
# from make_env import make_env
# import time
# import torch as T
# import numpy as np

# def obs_list_to_state_vector(observation):
#     state = np.array([])
#     for obs in observation:
#         state = np.concatenate([state, obs])
#     return state



# scenario = 'simple_adversary'
# env = make_env(scenario)
# n_agents = env.n
# actor_dims = []
# for i in range(n_agents):
#     actor_dims.append(env.observation_space[i].shape[0])
# critic_dims = sum(actor_dims)


# # action space is a list of arrays, assume each agent has same action space
# n_actions = env.action_space[0].n
# maddpg_agents = MADDPG(actor_dims, critic_dims, n_agents, n_actions, 
#                         fc1=64, fc2=64,  
#                         alpha=0.01, beta=0.01, scenario=scenario,
#                         chkpt_dir='tmp/maddpg/')


# maddpg_agents.load_checkpoint()
# obs = env.reset()

# for i in range(4):
#     env.render()

#     # print("observations", obs)
#     actions = maddpg_agents.eval_choose_action(obs)
#     # print("actions:", actions)
#     # print("")

#     # device = maddpg_agents.agents[0].actor.device
#     # agent = maddpg_agents.agents[0]

#     # print(obs)
#     # print(obs_list_to_state_vector(obs))

#     # states_ = T.tensor(obs_list_to_state_vector(obs), dtype=T.float).to(device)
#     # actions_ = T.tensor(actions, dtype=T.float).to(device)
    
#     # agents_actions = []
#     # for agent_idx, agent in enumerate(maddpg_agents.agents):
#     #     agents_actions.append(actions_[agent_idx])
#     # print(agents_actions)
#     # actions__ = T.cat([acts for acts in agents_actions], dim=1)

#     # test = agent.target_critic.forward(states_, actions__).flatten()
#     # print(test)

#     obs, reward, done, info = env.step(actions)
#     input("Press Enter to continue...")


# # import numpy as np
# # print(np.random.choice(20,7,replace=False))

# answer = False
# while not answer:
#     user_input = input("Max number of episodes reached, would you like to save the models? (y/n)")
#     if user_input.lower() == 'y':
#         print("yes")
#         answer = True
#     elif user_input.lower() == 'n':
#         pass
#         answer = True
#     else:
#         print("Invalid reply")

a = 0
import time


while True:
    try:
        a = a+1
    except KeyboardInterrupt:
        user_input = input("do you want to print")
        if user_input.lower()=="y":
            print(a)
        else:
            print("")
        time.sleep(1)
        break