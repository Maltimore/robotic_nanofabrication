import numpy as np


class Reward:
    def __init__(self, step_reward=0.5, success=10., fail=-10.):
        self.step_reward = step_reward
        self.success = success
        self.fail = fail

    def calculate(self, s1, s2, ruptures, successes):
        s1 = s1.reshape((-1, 1, 3))
        s2 = s2.reshape((-1, 1, 3))

        rupture_reward = np.isclose(s2, ruptures).all(axis=2).any(axis=1) * self.fail
        success_reward = np.isclose(s2, successes).all(axis=2).any(axis=1) * self.success

        reward = self.step_reward \
                 + rupture_reward \
                 + success_reward
        return reward
