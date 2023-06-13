from env import AshtaChammaEnv
from policies.random_policy import random_policy

if __name__ == "__main__":
    env = AshtaChammaEnv(random_policy)
    observation, _ = env.reset()

    done = False
    while not done:
        env.render()
        while True:
            try:
                move = int(input("Enter which piece you want to move (1-4): "))
            except:
                print("That's not an allowed move!")
                continue
            if env.action_space.contains(move - 1):
                break
            else:
                print("That's not an allowed move!")

        observation, reward, done = env.step(move - 1)
        print()

    print("Game over. Reward:", reward)
