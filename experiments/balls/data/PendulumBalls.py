import numpy as np
from experiments.balls.data.AbstractBalls import AbstractBalls


class PendulumBalls(AbstractBalls):
    STATE_DIM = 2

    def __init__(self,
                 n_balls,
                 img_size,
                 episode_length,
                 train_episodes,
                 test_episodes,
                 transition_noise_std=0.0,
                 dt=0.05,
                 sim_dt=1e-4,
                 mass=1.0,
                 lenght=1.0,
                 g=9.81,
                 friction=0.0,
                 seed=None,
                 scale_factor=0.8):
        """
        Also see super
        :param dt: difference between the time steps
        :param sim_dt: time difference for the internal simulator (should be smaller than dt)
        :param masses: masses of the links  (default 1 kg)
        :param lenghts: lenghts of the links (default 1 m)
        :param g: gravitational constant, (default 9.81 m/s^2)
        :param friction: friction for each link (default: no friction)
        """
        self.dt = dt
        self.sim_dt = sim_dt
        self.mass = mass
        self.length = lenght
        self.inertia = self.mass * (self.length**2) / 3
        self.g = g
        self.friction = friction
        self.scale_factor = scale_factor
        self.transition_noise_std = transition_noise_std
        super().__init__(n_balls, PendulumBalls.STATE_DIM, img_size, episode_length, train_episodes, test_episodes,
                         first_n_clean=5, seed=seed)

    def _initialize_state(self):
        """see super"""
        return np.array([self.random.uniform(-np.pi, np.pi), 0.0])

    def _transition_function(self, state):
        """see super"""
        nSteps = self.dt / self.sim_dt
        if nSteps != np.round(nSteps):
            print('Warning from Pendulum: dt does not match up')
            nSteps = np.round(nSteps)

        c = self.g * self.length * self.mass / self.inertia
        for i in range(0, int(nSteps)):
            velNew = state[1] + self.sim_dt * (c * np.sin(state[0]) - state[1] * self.friction)
            state = np.array([state[0] + self.sim_dt * velNew, velNew])
        if self.transition_noise_std > 0.0:
            state[1] += self.random.normal(loc=0.0, scale=self.transition_noise_std)
        return state

    def _get_task_space_states(self, states):
        """see super"""
        positions = np.zeros((self.episode_length, 2))
        positions[:, 0] += np.sin(states[:, 0])
        positions[:, 1] += np.cos(states[:, 0])
        # map to interval [-1, 1], i.e. the plotted region
        if self.scale_factor is not None:
            return positions * self.length * self.scale_factor
        else:
            return positions

    @property
    def has_task_space_velocity(self):
        return False

    @property
    def initial_state(self):
        return None

if __name__ == "__main__":
    data = PendulumBalls(n_balls=[25, 50],
                         img_size=[32, 32, 3],
                         episode_length=75,
                         train_episodes=2000,
                         test_episodes=1000,
                         transition_noise_std=0.1,
                         friction=0.1,
                         seed=42)

    train_obs, train_targets = data.train_observations, data.train_positions
    test_obs, test_targets = data.test_observations, data.test_positions
    np.savez_compressed("../pend_balls_first_clean.npz", train_obs=train_obs, test_obs=test_obs,
                        train_targets=train_targets, test_targets=test_targets)
