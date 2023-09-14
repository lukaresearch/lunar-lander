import gym
import torch
import numpy as np
from collections import deque
from Doubledqn_agent import Agent


episode = input('episode: ')
filename = 'Doubledqn' + episode
ModelFileName =  filename + '.pth'
outfile = open(filename + '.txt', 'w') 
Nepisode = int(episode)

env = gym.make('LunarLander-v2')
env.seed(0)
agent = Agent(state_size=8, action_size=4, seed=0)

def dqn(n_episodes=Nepisode, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break 
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        sc = np.mean(scores_window)

        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, sc), end="")
        print("%d %.2f" %(i_episode, sc), file = outfile)

        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
   
    torch.save(agent.qnetwork_local.state_dict(), ModelFileName)
    print('\nTraining %d epsisodes in file %s' %(Nepisode, ModelFileName))

    return scores


scores = dqn()
env.close()