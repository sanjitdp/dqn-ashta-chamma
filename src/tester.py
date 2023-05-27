from env import AshtaChammaEnv

if __name__ == "__main__":
    env = AshtaChammaEnv()
    env.reset()
    env.step((1, 2))
    env.switch_player()
    env.step((2, 4))
    env.switch_player()
    print(env.step((1, 2)))
    env.render()
