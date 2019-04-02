import gym
import numpy as np

env = gym.make('FrozenLake-v0')


QTable = np.zeros([env.observation_space.n,env.action_space.n])

alpha = .80
gamma = .95
NumEpisodes = 2000

RewardsList = []
for i in range(NumEpisodes):
    CState = env.reset()
    SumReward = 0
    d = False
    j = 0
    while j < 99:
        j+=1
        Action = np.argmax(QTable[CState,:] + np.random.randn(1,env.action_space.n)*(1./(i+1)))
        NState,Rewards,d,_ = env.step(Action)
        QTable[CState,Action] = QTable[CState,Action] + alpha*(Rewards + gamma*np.max(QTable[NState,:]) - QTable[CState,Action])
        SumReward += Rewards
        CState = NState
        if d == True:
            break
    
    RewardsList.append(SumReward)

print ("Score: " +  str(sum(RewardsList)/NumEpisodes))

print ("Final Q-Table Values")
print (QTable)
