import numpy as np


class SpringDamper:

    def __init__(self, sigma_trans=0.01, sigma_obs=0.1, m=100, b=0.99, dt=1, seed=0):
        self._m = m
        self._b = b
        self._dt = dt
        self._sigma_trans = sigma_trans
        self._sigma_obs = sigma_obs
        self._rng = np.random.RandomState(seed)

    def run(self, initial_state, num_steps):

        states = np.zeros((num_steps + 1, 4))
        states[0] = initial_state
        for i in range(num_steps):
            states[i+1] = states[i] @ self.transition_matrix.T
            states[i+1] += np.random.multivariate_normal(np.zeros(4), self.transition_noise_covar)
        obs = states @ self.observation_matrix.T
        obs += np.random.multivariate_normal(mean=np.zeros(2), cov=self.observation_noise_covar, size=len(obs))
        return obs, states

    @property
    def transition_matrix(self):
        """Transition matrix of the linear ball tracking - copied from original implementation by T. Harnooja"""
        return np.array([[1, 0.0, self._dt, 0.0],
                         [0.0, 1, 0.0, self._dt],
                         [-self._dt / self._m, 0.0, self._b, 0.0],
                         [0.0, -self._dt / self._m, 0.0, self._b]])

    @property
    def observation_matrix(self):
        return np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0]])

    @property
    def transition_noise_covar(self):
        return (self._sigma_trans ** 2) * np.eye(4)

    @property
    def observation_noise_covar(self):
        return (self._sigma_obs ** 2) * np.eye(2)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    sd = SpringDamper(sigma_obs=0.5)

    np.random.seed(1)
    init_state = np.concatenate([np.random.uniform(low=-0.2, high=0.2, size=2),
                                 np.random.uniform(low=-0.2, high=0.2, size=2)], 0)

    obs, states = sd.run(init_state, 200)

    plt.plot()
    for i in range(4):
        plt.subplot(4, 1, i+1)
        plt.plot(states[:, i], c="blue")
        if i < 2:
            plt.scatter(np.arange(0, len(obs), 1), obs[:, i], c="green")
    plt.show()






