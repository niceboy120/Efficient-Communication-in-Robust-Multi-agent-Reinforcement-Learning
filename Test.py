from MADDPG.maddpg import MADDPG
from make_env import make_env
import time
import torch as T
import numpy as np

def obs_list_to_state_vector(observation):
    state = np.array([])
    for obs in observation:
        state = np.concatenate([state, obs])
    return state



scenario = 'simple_adversary'
env = make_env(scenario)
n_agents = env.n
actor_dims = []
for i in range(n_agents):
    actor_dims.append(env.observation_space[i].shape[0])
critic_dims = sum(actor_dims)


# action space is a list of arrays, assume each agent has same action space
n_actions = env.action_space[0].n
maddpg_agents = MADDPG(actor_dims, critic_dims, n_agents, n_actions, 
                        fc1=64, fc2=64,  
                        alpha=0.01, beta=0.01, scenario=scenario,
                        chkpt_dir='MADDPG/tmp/maddpg/')


maddpg_agents.load_checkpoint()
obs = env.reset()
sequence = [obs]

for i in range(2):
    # env.render()

    print("obs: \n", obs)

 
    
    actions = []
    actions_env = []

    for agent_idx, agent in enumerate(maddpg_agents.agents):
        device = maddpg_agents.agents[agent_idx].target_actor.device

        agent_state = T.tensor([obs[agent_idx]], dtype=T.float).to(device)
        action = agent.target_actor.forward(agent_state)

        actions.append(action)
        actions_env.append(action.detach().cpu().numpy()[0])

    mu = T.cat([acts for acts in actions], dim=1)

    Q_all = []
    for agent_idx, agent in enumerate(maddpg_agents.agents):
        device = maddpg_agents.agents[agent_idx].target_critic.device
        Q = agent.target_critic.forward(T.tensor([obs_list_to_state_vector(obs)], dtype=T.float).to(device), mu).flatten()
        Q_all.append(Q.detach().cpu().numpy()[0])

    obs, reward, done, info = env.step(actions_env)
    sequence.append(obs)
    print("")
    print("Q_all: ", Q_all, Q_all[1:])
    print("")
    # input("Press Enter to continue...")

# sequence = np.array(sequence)
print(sequence)
print("")
seq1 = np.concatenate((sequence[0][0], sequence[0][1], sequence[0][2]))
seq2 = np.concatenate((sequence[1][0], sequence[1][1], sequence[1][2]))

print(np.linalg.norm(seq1-seq2))


