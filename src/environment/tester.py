from env import AshtaChammaEnv
from policies.random_policy import random_policy

if __name__ == "__main__":

    env = AshtaChammaEnv(opponent_policy=random_policy)
    env.reset()
    
    done = False
    while not done:
        observation, reward, done = env.step(random_policy(env))
        env.render()
