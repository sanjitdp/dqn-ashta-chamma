from env import AshtaChammaEnv
from policies.model_policy import model_policy

if __name__ == "__main__":
    env = AshtaChammaEnv(model_policy)
    observation, _ = env.reset()

    done = False
    while not done:
        env.render()
        while True:
            try:
                move = int(input("Enter which piece you want to move (1-4): "))
            except ValueError:
                print("That's not an allowed move!")
                continue
            if move < 1 or move > env.NUM_PIECES:
                print("That's not an allowed move!")
            elif env.is_illegal_move(move - 1) and not all(
                env.is_illegal_move(i) for i in range(env.NUM_PIECES)
            ):
                print("That's an illegal move, but you have other legal options.")
            elif env.action_space.contains(move - 1):
                break
            else:
                print("That's not an allowed move!")

        observation, reward, done = env.step(move - 1)
        print()

    print("Game over. Reward:", reward)
