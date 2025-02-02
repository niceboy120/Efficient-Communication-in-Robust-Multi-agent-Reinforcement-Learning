import numpy as np
from MPE.multiagent.core import World, Agent, Landmark
from MPE.multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self, environment):
        world = World(environment)
        # set any world properties first
        world.dim_c = 2
        num_good_agents = 1
        num_adversaries = 2
        num_agents = num_adversaries + num_good_agents
        num_landmarks = 0
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.adversary = True if i < num_adversaries else False
            agent.size = 0.075 if agent.adversary else 0.05
            agent.accel = 3.0 if agent.adversary else 4
            #agent.accel = 20.0 if agent.adversary else 25.0
            agent.max_speed = 0.546 if agent.adversary else 0.546
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = True
            landmark.movable = False
            landmark.size = 0.2
            landmark.boundary = False
        # make initial conditions
        self.reset_world(world)
        return world



    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.85, 0.35]) if not agent.adversary else np.array([0.85, 0.35, 0.35])
            # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_rot = np.random.uniform(0, np.pi, 1)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            if not landmark.boundary:
                landmark.state.p_pos = np.random.uniform(-0.9, +0.9, world.dim_p)
                landmark.state.p_vel = np.zeros(world.dim_p)


    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        if agent.adversary:
            collisions = 0
            for a in self.good_agents(world):
                if self.is_collision(a, agent):
                    collisions += 1
            return collisions
        else:
            return 0


    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]


    def reward(self, agent, world, reward_mode):
        # Agents are rewarded based on minimum agent distance to each landmark
        if agent.adversary:
            main_reward, collisions = self.adversary_reward(agent, world, reward_mode)  
            return main_reward, collisions
        else:
            main_reward, collisions = self.agent_reward(agent, world)
            return main_reward, collisions

    def agent_reward(self, agent, world):
        # Agents are negatively rewarded if caught by adversaries
        rew = 0
        shape = True
        adversaries = self.adversaries(world)
        if shape:  # reward can optionally be shaped (increased reward for increased distance from adversary)
            dist = []
            for adv in adversaries:
                dist.append(0.2 * np.sqrt(np.sum(np.square(agent.state.p_pos - adv.state.p_pos))))
            rew += min(dist)
        if agent.collide:
            for a in adversaries:
                if self.is_collision(a, agent):
                    rew += -10

        # agents are penalized for exiting the screen, so that they can be caught by the adversaries
        def bound(x):
            if x < 0.8:
                return 0
            if x < 1.0:
                return (x - 0.8) * 100
            return min(np.exp(3 * x ), 40)
            # else:
                # return 50
        for p in range(world.dim_p):
            x = abs(agent.state.p_pos[p])
            rew -= bound(x)

        return rew, 0

    def adversary_reward(self, agent, world, reward_mode=None):
        # Adversaries are rewarded for collisions with agents
        rew = 0
        shape = True
        agents = self.good_agents(world)
        adversaries = self.adversaries(world)
        if shape:  # reward can optionally be shaped (decreased reward for increased distance from agents)
            dist = []
            for adv in adversaries:
                dist.append(min([np.sqrt(np.sum(np.square(a.state.p_pos - adv.state.p_pos))) for a in agents]))
            rew += -min(dist)

        for adv in adversaries:
            if adv.collide:
                for ag in agents: 
                    if self.is_collision(ag, adv):
                        rew += 1

        pos = None
        for adv in adversaries:
            pos_old = pos
            pos = adv.state.p_pos
        
        rew += -np.sqrt(np.sum(np.square(pos+pos_old)))            

        collisions = [0,0,0]
        return rew, collisions

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            if not entity.boundary:
                entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # communication of all other agents
        comm = []
        other_pos = []
        other_vel = []
        for other in world.agents:
            if other is agent: continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
            other_pos.append(other.state.p_rot - agent.state.p_rot)
            if not other.adversary:
                other_vel.append(other.state.p_vel)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + [agent.state.p_rot] + entity_pos + other_pos + other_vel)


