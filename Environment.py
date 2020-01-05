class Environment:
    def __init__(self, env):
        self.environment = env

    def run(self, selector, train, verbose, render):
        state = self.environment.reset()
        reward = 0
        selected, _, std = selector.select(state)
        while True:
            if render:
                self.environment.render()
            action = selector.act(state)
            new_state, current_reward, done, info = self.environment.step(action)

            if done:
                new_state = None

            if train:
                selector.observe((state, action, current_reward, new_state))
                selector.replay()

            state = new_state
            reward += current_reward
            if reward < -1000:
                break
            if done:
                break
        if verbose:
            print("Reward achieved: ", reward)
        return selected, reward, std
