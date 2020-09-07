import numpy as np
from experiments.balls.data.AbstractBalls import AbstractBalls
import _DoubleLinkForwardModel as DoubleLink

class DoubleLinkBalls(AbstractBalls):
    STATE_DIM = 4

    def __init__(self,
                 n_balls,
                 img_size,
                 episode_length,
                 train_episodes,
                 test_episodes,
                 dt=0.05,
                 sim_dt=1e-4,
                 masses=None,
                 lenghts=None,
                 g=9.81,
                 friction=None,
                 seed=None,
                 scale_factor=None):
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
        self.masses = masses if masses is not None else np.ones(2)
        self.lengths = lenghts if lenghts is not None else np.ones(2)
        self.inertias = self.masses * (self.lengths**2) / 3
        self.g = g
        self.friction = friction if friction is not None else np.zeros(2)
        self.scale_factor = scale_factor
      #  print("scale_factor: ", scale_factor)
        super().__init__(n_balls, DoubleLinkBalls.STATE_DIM, img_size, episode_length, train_episodes, test_episodes,
                         first_n_clean=5, seed=seed)

    def _initialize_state(self):
        """see super"""
        p1, p2 = [self.random.uniform(-np.pi, np.pi) for _ in range(2)]
        v1, v2 = np.zeros(2)#[self.random.uniform(-1, 1) for _ in range(2)]
        return np.array([p1, v1, p2, v2])

    def _transition_function(self, state):
        """see super"""
        state = np.expand_dims(state, 0)
        actions = np.zeros((state.shape[0], 2))
        result = np.zeros((state.shape[0], 6))
        DoubleLink.simulate(state, actions, self.dt, self.masses, self.lengths, self.inertias,
                            self.g, self.friction, self.sim_dt, 0, np.zeros(4), np.zeros(4), result)
        return result[0, :4]


    def _get_task_space_states(self, states):
        """see super"""
        positions = np.zeros((self.episode_length, 2))
        for i in range(2):
            positions[:, 0] += np.sin(states[:, 2 * i]) * self.lengths[i]
            positions[:, 1] += np.cos(states[:, 2 * i]) * self.lengths[i]
        # map to interval [-1, 1], i.e. the plotted region
        if self.scale_factor is not None:
            return positions * self.scale_factor
        else:
            return positions / np.sum(self.lengths)

    @property
    def has_task_space_velocity(self):
        return False

    @property
    def initial_state(self):
        return None

if __name__ == "__main__":
    data = DoubleLinkBalls(n_balls=3,
                           img_size=[64, 64, 3],
                           episode_length=100,
                           train_episodes=3,
                           test_episodes=3)
    #pos = np.clip((data.train_positions * 32).astype(np.int) + 32, a_min=0, a_max=63)

    #img_with_path = data.train_observations[0]
    #img_with_path[:, pos[0, :, 1], pos[0, :, 0], 0] = 0 #np.arange(start=55, stop=255, step=2, dtype=np.uint8)
    #img_with_path[:, pos[0, :, 1], pos[0, :, 0], 1] = np.arange(start=55, stop=255, step=2, dtype=np.uint8)
    #img_with_path[:, pos[0, :, 1], pos[0, :, 0], 2] = np.arange(start=255, stop=55, step=-2, dtype=np.uint8)

    #import matplotlib.pyplot as plt
    #plt.imshow(img_with_path[0], interpolation="none")
    #plt.show()

    data.images_to_vid(data.train_observations[0], "/home/philipp/projects/Balls/dummy.avi")
