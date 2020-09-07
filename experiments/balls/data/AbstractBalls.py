import abc
import numpy as np
from PIL import Image
from PIL import ImageDraw


class AbstractBalls(abc.ABC):
    """Abstract Superclass to generate data for Ball tracking task"""
    TRACK_BALL_COLOR = np.array([255, 0, 0], dtype=np.uint8)

    def __init__(self,
                 n_occlusion_balls,
                 state_dim,
                 img_size,
                 episode_length,
                 train_episodes,
                 test_episodes,
                 first_n_clean,
                 seed):
        """
        Creates new dataset
        :param n_occlusion_balls: number of occlusion balls in the image (i.e. total number is this + 1 (i.e. the ball
         to track), (if this is a negative number -n, the number of balls is sampled  between 0 and n. A new number is
        sampled for each sequence)
        :param state_dim: dimensionality of the state (given by subclasses)
        :param img_size: size of the images to generate
        :param episode_length: length of the sequences that will be generated
        :param train_episodes: number of sequences that will be generated
        :param first_n_clean: the ball to track will be rendered on top for those first n images
        :param seed: seed for the random number generator
        """

        self.random = np.random.RandomState(seed)
        self.n_balls = n_occlusion_balls
        self.state_dim = state_dim
        self.first_n_clean = first_n_clean
        self.img_size = img_size
        self._img_size_internal = [128, 128, 3]
        #generate one more - first is initial
        self.episode_length = episode_length + 1

        self._train_images, self._train_states = self._simulate_data(train_episodes)
        #self._train_visibility = self.compute_visibility(self._train_images)

        self._test_images, self._test_states = self._simulate_data(test_episodes)
        #self._test_visibility = self.compute_visibility(self._test_images)

    def _simulate_data(self, number_of_episodes):
        """
        Simulates the dataset
        :return: images (observations) and task_space_states (positions (+ velocity if 'has_task_space_velocity')) for the
        ball to track
        """

        images = np.zeros([number_of_episodes, self.episode_length] + self.img_size, dtype=np.uint8)
        ts_dim = 4 if self.has_task_space_velocity else 2
        task_space_states = np.zeros([number_of_episodes, self.episode_length, ts_dim])

        for i in range(number_of_episodes):
            # +1 since randint samples from an interval excluding the high value
            if isinstance(self.n_balls, tuple) or isinstance(self.n_balls, list):
                n_balls = self.random.randint(low=self.n_balls[0], high=self.n_balls[1])
            else:
                n_balls = self.n_balls if self.n_balls > 0 else self.random.randint(low=1, high=-self.n_balls + 1)
            print(i, n_balls)
            track_state = np.zeros([self.episode_length, self.state_dim])
            occ_states = np.zeros([self.episode_length, n_balls, 4])

            track_state[0, :] = self._initialize_state()
            occ_states[0, :] = self._initialize_states_occlusion(n_balls)

            for j in range(1, self.episode_length):
                track_state[j] = self._transition_function(track_state[j - 1])
                occ_states[j] = self._transition_function_occlusion(occ_states[j - 1])
            track_ts = self._get_task_space_states(track_state)
            occ_ts = occ_states[..., :2]
            images[i] = self._render(track_ts[..., :2], occ_ts)
            task_space_states[i] = track_ts
        return images, task_space_states

    def _initialize_states_occlusion(self, n_balls):
        """ see super"""
        init_state = self.random.rand(n_balls, 4) * 2 - 1
        init_state[:, 2:] *= 0.1
        return init_state

    def _transition_function_occlusion(self, state):
        """see super"""
        # state contains the states as row vectors (stacked to a matrix), so should the new states,
        # hence we can compute s_{t+1}^T = s_t^T A^T instead of s_{t+1} = A s_t
        next_state = np.matmul(state, np.transpose(self._transition_matrix))
        #next_state[:, 2:] += self.dyn_sigma * self.random.randn(n_balls, 2)
        return next_state

    @property
    def _transition_matrix(self):
        """Transition matrix of the linear ball tracking - copied from original implementation by T. Harnooja"""
        return np.array([[1,     0,     1,    0    ],
                         [0,     1,     0,    1    ],
                         [-0.01, 0,     0.99, 0    ],
                         [0,     -0.01, 0,    0.99]])


    def _render(self, tb_task_space_state, ob_task_space_states):
        """
        Creates images out of positions
        :param task_space_states: Batches of sequences of the task space states of all balls as an
               [episode_length x number_of_balls x task space dim]
        :return: sequence of images
        """

        images = np.zeros([self.episode_length] + self.img_size, dtype=np.uint8)
        ob_radii = self.random.randint(low=5, high=10, size=ob_task_space_states.shape[1])
        # Those magic numbers where chosen to match the original ball data implementation by T. Haarnoja
        ob_colors = self.random.randint(low=0, high=255, size=[ob_task_space_states.shape[1], 3])
        for i in range(self.episode_length):
            img = Image.new('RGB', self._img_size_internal[:2])
            draw = ImageDraw.Draw(img)
            if i >= self.first_n_clean:
                self._draw_ball(tb_task_space_state[i], 5, AbstractBalls.TRACK_BALL_COLOR, draw)
            for j in range(ob_task_space_states.shape[1]):
                self._draw_ball(ob_task_space_states[i, j], ob_radii[j], ob_colors[j], draw)
            if i < self.first_n_clean:
                self._draw_ball(tb_task_space_state[i], 5, AbstractBalls.TRACK_BALL_COLOR, draw)
            img = img.resize(self.img_size[:2], Image.ANTIALIAS)
            images[i] = np.array(img)
        return images

    def _draw_ball(self, pos, rad, col, draw):
        x, y = pos
        x = np.int64((x + 1) * self._img_size_internal[0] / 2)
        y = np.int64((y + 1) * self._img_size_internal[1] / 2)
        draw.ellipse((x - rad, y - rad, x + rad, y + rad), fill=tuple(col), outline=tuple(col))

    def images_to_vid(self, images, filename):
        """
        Generates video out of sequence of images
        :param images: sequence of images
        :param filename: filname to save video under (including path)
        :return:
        """
        import matplotlib.animation as anim
        import matplotlib.pyplot as plt

        assert len(images) == self.episode_length - 1, "Length of given sequence does not match sequence length, something wrong"
        fig = plt.figure()
        axes = plt.gca()
        axes.get_xaxis().set_visible(False)
        axes.get_yaxis().set_visible(False)
        img_dummy = axes.imshow(images[0], interpolation='none')
        ani = anim.FuncAnimation(fig, lambda n: img_dummy.set_data(images[n]), len(images))
        writer = anim.writers['ffmpeg'](fps=10)
        ani.save(filename, writer=writer, dpi=100)
        plt.close()

    def compute_visibility(self, observations):
        raise NotImplementedError("Addapt to new antialiasing")
        """Counts how many pixels of the ball are visible in each image
        :param observations: batch of sequences of images"""
        color_reshape = np.reshape(AbstractBalls.TRACK_BALL_COLOR, [1, 1, 1, 1, 3])

        # this subtraction also works in uint8 (its not a reasonable value but it is zero if and only if it should be)
        observations = observations - color_reshape        # subtract ball color from images - count how many are zero

        # count how many are zeros there are for each image
        pixel_is_color = np.all(observations == 0, axis=-1)
        values = np.sum(pixel_is_color, (2, 3))

        return values

    @property
    def train_positions(self):
        """
        The true positions of the ball from t=1
        :return:
        """
        return self._train_states[:, 1:, :2]

    @property
    def test_positions(self):
        return self._test_states[:, 1:, :2]

    @property
    def train_observations(self):
        """
        The observations of the ball from t=1
        :return:
        """
#        print(np.max(self._images), np.average(self._images))
        return self._train_images[:, 1:]

    @property
    def test_observations(self):
        return self._test_images[:, 1:]

    @property
    def initial_state(self):
        """The initial latent state at t=0 (may not be given)"""
        raise NotImplementedError("Initial latent state unknown and can not be given!")


    @abc.abstractmethod
    def _initialize_state(self):
        """Sample initial states for all balls"""
        raise NotImplementedError("State Initialization not implemented")

    @abc.abstractmethod
    def _get_task_space_states(self, states):
        """
        Map states to task space states, needs to be capable to handle sequences of batches (of n balls)
        :param states: states in not task space (e.g. joint space for robotic system or n-link pendulum
        :return: states in task space (location of center of the ball in the image, if 'has_task_space_velocity'
        this also needs to return the velocity of the ball (in the image)
        """
        raise NotImplementedError("Task Space Mapping not implemented")

    @abc.abstractmethod
    def _transition_function(self, state):
        """
        Maps from current state to next state, needs to be capable of handling batches (of n balls)
        :param state: current state
        :return: next state
        """
        raise NotImplementedError("Transition Function not implemented")

    @property
    def has_task_space_velocity(self):
        raise NotImplementedError("Has task space velocity not implemented ")