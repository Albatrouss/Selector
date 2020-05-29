class Environment:
    def __init__(self, env, name=""):
        self.environment = env
        self.name = name

    def run(self, selector, train, verbose, render):
        state = self.environment.reset()
        reward = 0
        selected, log, std = selector.select(state)
        while True:
            if render:
                self.environment.render()
            action, std = selector.act(state)
            new_state, current_reward, done, info = self.environment.step(action)

            if done:
                new_state = None

            if train:
                selector.observe((state, action, current_reward, new_state))
                selector.replay()

            state = new_state
            reward += current_reward
            if reward < -1000:#to avoid getting stuck in LL
                break
            if done:
                break
        if verbose:
            print("Steps: {}, Reward achieved: {}".format(selector.selected_agent.steps, reward))
        return selected, reward, std, log
