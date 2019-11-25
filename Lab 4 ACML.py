import numpy as np
import gym
import matplotlib.pyplot as plt

#to allow the agent to train for a max of 100000 steps
gym.envs.register(

    id='MountainCarMyEasyVersion-v0',

    entry_point='gym.envs.classic_control:MountainCarEnv',

    max_episode_steps=10000,      # total number of episodes agent plays during training 
    )

env = gym.make('MountainCarMyEasyVersion-v0')
action_space_size = env.action_space.n

max_steps_per_episode = 100
slow = False
reward = None
steps_to_goal = []
number_states = 40   # discretize state space
epsilon = 0.1     # greedy action picking
exploration_rate = 1   # initialize
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate = 0.01 # rate the exploration rate decays
rewards_all_episodes = 0
discount_rate = 1    # no discount
alpha = 0.1  # learning rate
#mode 1 for sarsa 0 for Q-learning
mode = 0
q_table = np.zeros((number_states, number_states, 3))  # initialize q-table (3 actions: left, stay,right)

def observation_to_state(env, observation, number_states):
    # map an observation to state
    environment_low = env.observation_space.low
    environment_high = env.observation_space.high

    # position and velocity divided into 40 intervals
    environment_dx = (environment_high - environment_low) / number_states
    # position = observation[0] ;  volecity = observation[1]
    p = int((observation[0] - environment_low[0])/environment_dx[0])
    v = int((observation[1] - environment_low[1])/environment_dx[1])
    return p, v


for episode in range(10000):        #single episode

    state = env.reset() #starting state
    timesteps, G, rewards_current_episode = 0,0,0 # initialize 
    policy = np.argmax(q_table, axis=2)    # highest q value in the current state
    
    start = True

    finished = False

    
    while not finished:            # single step
        p, v = observation_to_state(env, state, number_states)

        exploration_rate_threshold = np.random.uniform(0, 1) #explore or exploit in this step
        if slow: env.render()
        
        if(start):
            action = env.action_space.sample() #random action at the start
            start = False

        if exploration_rate_threshold < exploration_rate:
            action = np.random.choice(action_space_size) #explore
            #action = env.action_space.sample() # explore
            
        else:
            action = policy[p][v]   #action following the policy / exploit

        #next observation according to the action
        new_state, reward, finished, info = env.step(action)
        #new state given the action
        p_, v_ = observation_to_state(env, new_state, number_states)
        #new action according to the new state and the policy for that state needed for SARSA
        new_action = policy[p_][v_]

        if mode==1:
            #SARSA update: Q(s,a) + alpha*[r + gamma*Q(s',a') - Q(s, a)]
            q_table[p][v][action] = q_table[p][v][action] + alpha * (reward +  (discount_rate * q_table[p_][v_][new_action] - q_table[p][v][action]))
        else:
            #Q update: Q(s,a) + alpha*[r + gamma*max(Q(s',a')) - Q(s, a)]
            q_table[p][v][action] = q_table[p][v][action] + alpha * (reward +  (discount_rate * np.max(q_table[p_][v_]) - q_table[p][v][action]))
        
        #policy update
        policy = np.argmax(q_table, axis=2)
        #cumulative reward
        #-1 for each time step 
        rewards_current_episode += reward
        #to continue from the state we are at now
        state = new_state
        timesteps+=1

        if slow: print (state)
        if slow: print (rewards_current_episode)
        if slow: print (finished, timesteps, G)
        #to see how many steps it took for each episode    
        if finished :                                   #episode over
            steps_to_goal.append(timesteps)           
    #to see the progress   
    if episode % max_steps_per_episode == 0:   # 100 steps within an episode
        
        print ('Iteration:' , episode, "Episode terminated after ", timesteps, "steps.", "Accumulated reward: ", rewards_current_episode)
    #Exploration reate decay
    exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate*episode)
#getting the max values of the q-tables and printing them
q_max = np.amax(q_table, axis=2)
plt.imshow(q_max, origin='lower', label='accumulated reward')
plt.colorbar()
plt.show()
#Printing the best policy
print (policy)
plt.plot(steps_to_goal) #per episode
plt.show()
