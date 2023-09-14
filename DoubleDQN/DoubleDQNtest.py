import gym
import torch
from Doubledqn_agent import Agent

Ntest = 5

episode = input('episode: ')
filename = 'Doubledqn' + episode
ModelFileName =  filename + '.pth'

StartSeed = 900

env = gym.make('LunarLander-v2')
env.seed(StartSeed)
agent = Agent(state_size=8, action_size=4, seed=0)
MaxStep = 500

agent.qnetwork_local.load_state_dict(torch.load(ModelFileName))
TotalScore = 0.
for i in range(Ntest):
    state = env.reset()
    score = 0.
    for j in range(MaxStep):
        action = agent.act(state)
        state, reward, done, _ = env.step(action)
        score += reward
        env.render()                
        if done:
            break 
    TotalScore += score

    print('\rEpisode {}\tScore: \t{:.2f}'.format(i, score))

print('AverageScore\t\t%.2f' %(TotalScore/Ntest)            )
env.close()