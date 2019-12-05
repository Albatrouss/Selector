class Environment:
    def __init__(self, env):
        self.environment = env

    def run(self, selector, train, verbose, render):
        state = self.environment.reset()
        Reward = 0
        selector.select(state)
        while True:
            if render:
                self.environment.render()
            action = selector.act(state)
            newstate, currentreward, done, info = self.environment.step(action)

            if done:
                newstate = None

            if train:
                selector.observe((state, action, currentreward, newstate))
                selector.replay()

            state = newstate
            Reward += currentreward

            if done:
                break
        if verbose:
            print("Reward achieved: ", Reward)