from env import AshtaChammaEnv

if __name__ == "__main__":
    env = AshtaChammaEnv()

    # test capture
    env.reset()
    env.step((1, 2))
    env.step((2, 4))
    env.step((1, 2))
    env.step((3, 2))
    env.render()

    # test win condition
    env.reset()
    env.step((0, 4))
    env.step((0, 4))
    env.step((0, 4))
    env.step((0, 4))
    env.step((0, 4))
    env.step((0, 4))
    env.step((0, 4))
    env.step((0, 4))

    env.step((0, 4))
    env.step((0, 4))
    env.step((0, 4))
    env.step((0, 4))
    env.step((0, 4))
    env.step((0, 4))
    env.step((0, 4))
    env.step((0, 4))

    env.step((0, 4))
    env.step((0, 4))
    env.step((0, 4))
    env.step((0, 4))
    _, reward, done = env.step((0, 4))
    print(reward, done) # should print "0 False"
    env.step((0, 4))
    _, reward, done = env.step((0, 4))
    print(reward, done) # should print "1 True"
    env.render()
