import gym
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Reshape
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory


ENV_NAME = 'FrozenLake-v0'

env = gym.make(ENV_NAME)
np.random.seed(1)
env.seed(1)
Actions = env.action_space.n

model = Sequential()
model.add(Embedding(16, 4, input_length=1))
model.add(Reshape((4,)))
print(model.summary())

memory = SequentialMemory(limit=10000, window_length=1)
policy = BoltzmannQPolicy()
Dqn = DQNAgent(model=model, nb_actions=Actions,
               memory=memory, nb_steps_warmup=500,
               target_model_update=1e-2, policy=policy,
               enable_double_dqn=False, batch_size=512
               )
Dqn.compile(Adam())


Dqn.fit(env, nb_steps=1e5, visualize=False, verbose=1, log_interval=10000)

Dqn.save_weights('dqn_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

Dqn.test(env, nb_episodes=20, visualize=False)



