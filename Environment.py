class Environment:
    def __init__(self, env):
        self.environment = env

    def run(self, selector, verbose):
        state = self.environment.reset()
        Reward = 0

        while True:
            self.environment.render()
            action = selector.act(state)

            newstate, currentreward, done, info = self.environment.step(action)

            if done:
                newstate = None

            selector.observe((state, action, currentreward, newstate))
            selector.replay()

            state = newstate
            Reward += currentreward

            if done:
                break
        if verbose:
            print("Reward achieved: ", Reward)