import numpy as np
import os
import logging
import ast
import warnings


logger = logging.getLogger('experiencebuf')
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(msecs)d - %(levelname)s - %(message)s')
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
handler.setFormatter(formatter)
logger.addHandler(handler)


# note that we needed to implement a custom experience buffer unlike the one used
# in popular RL frameworks like rllib. This is because we need to be able to access
# and delete entire episodes if necessary. So we need to keep track of which sample
# belongs to which episode.
class ExperienceBuffer:
    def __init__(self, experience_path, keep_n_episodes, buffer_dtype):
        """__init__

        :param experience_path: string
            path to experience folder
        :param keep_n_episodes: scalar
            number of episodes to keep in memory
        :param buffer_dtype: list of 2-tuples, first element string, second numpy dtype
            mapping from experience type to dtype
        """
        self.logger = logger  # global logger of this module

        self.path = experience_path
        self.keep_n_episodes = keep_n_episodes
        self.buffer_dtype = buffer_dtype

        self.accepted_episodes_path = os.path.join(self.path, 'accepted_episodes.txt')
        self.accepted_episodes_indices = []
        self.current_episode_container = []
        self.episode_buffers = []
        self.buffer = np.empty(shape=(0,),
                               dtype=self.buffer_dtype)

        self.load()

    def __len__(self):
        return self.buffer.shape[0]

    def add(self, experience):
        """add

        :param experience: list of len 6
            - state_1
            - action idx
            - reward
            - state_2
            - TD error (typically initialized to high value)
            - terminal state (bool)
        """
        self.current_episode_container.append(experience)

    def finish_episode(self):
        """finish_episode
        Indicates to the Experience buffer that the current episode is finished,
        all new added samples will be part of a next episode"""
        current_episode_buffer = self._make_buffer(self.current_episode_container)
        self.episode_buffers.append(current_episode_buffer)
        self.accepted_episodes_indices.append(len(self.episode_buffers) - 1)
        self.combine_episode_buffers()
        self.current_episode_container = []

    def _make_buffer(self, episode):
        """_make_buffer
        Creates an array from a list whose entries are steps in the episode

        :param episode: list of episode steps
        """
        buffer_ = np.empty(shape=(len(episode),),
                           dtype=self.buffer_dtype)
        for i in range(len(episode)):
            buffer_[i] = tuple(episode[i])
        return buffer_

    def combine_episode_buffers(self):
        """combine_episode_buffers
        Combines the buffers of accepted episodes into one big buffer and assigns it to
        self.buffer. It also cuts out old episode buffers if there are more
        episodes saved than allowed by self.keep_n_episodes"""
        # determine which buffer indices need to be combined
        buffer_indices_to_combine = []
        n_combined_buffers = 0
        total_buffer_length = 0
        for i in range(len(self.episode_buffers) - 1, -1, -1):
            if n_combined_buffers <= self.keep_n_episodes:
                if self.episode_buffers[i] is not None:
                    buffer_indices_to_combine.insert(0, i)
                    total_buffer_length += len(self.episode_buffers[i])
                    n_combined_buffers += 1
            else:
                self.episode_buffers[i] = None

        # create a big array to put the episode buffers into such that
        # there's no copy on concatenate
        self.buffer = np.empty(shape=(total_buffer_length,),
                               dtype=self.buffer_dtype)
        idx = 0
        for buffer_idx in buffer_indices_to_combine:
            current_buffer_length = len(self.episode_buffers[buffer_idx])
            self.buffer[idx:idx + current_buffer_length] = self.episode_buffers[buffer_idx]
            self.episode_buffers[buffer_idx] = self.buffer[idx:idx + current_buffer_length]
            idx += current_buffer_length

        # find uniqe states
        self.unique_positions = []
        for i in range(self.buffer.shape[0]):
            new_state = self.buffer[i]
            if not any((new_state['s1'][:3] == x).all() for x in self.unique_positions):
                self.unique_positions.append(new_state['s1'][:3])
        self.unique_positions = np.array(self.unique_positions)

    def delete_episode(self, n):
        logger.info('Exp: Deleting episode  {} of {}'.format(n, len(self.episode_buffers) - 1))
        if n > len(self.episode_buffers) - 1:
            raise ValueError('Was asked to forget episode {} '.format(n) +
                             'but current_episode is only {}'.format(len(self.episode_buffers) - 1))
        elif n < 0:
            logger.warning('Was asked to delete episode {} which is < 0! Ignoring.'.format(n))
        else:
            self.accepted_episodes_indices[n] = None
            self.episode_buffers[n] = None
            self.combine_episode_buffers()

    def sample(self, n, sample_type='prioritized', return_indices=False):
        """sample

        :param n: scalar
            number of steps sampled
        :param sample_type: string
            one of:
                'random': samples drawn uniformly
                'last_n': last n samples
                'prioritized': prioritizd according to TD error of last training step
        :param return_indices: bool
            whether to return the sampled indices

        :returns: array of shape (n, 6)
            the columns contain in order:
            s1: list of shape (n_state_items,)
                with previous state
            a1: scalar
                actions of previous states
            r1: scalar
                returns of actions of previous states
            s2: array of shape (n_state_items,)
                next state
            td: Temporal-Difference error
            t2: bool
                True if S2 is terminal state
        """
        buffersize = len(self)
        # input checks
        if buffersize == 0:
            raise Exception('No experience yet, cannot sample!')
        if len(self.current_episode_container) > 0:  # unfinished episode
            warnings.warn('Sampling from the experience buffer while ' +
                          'having unfinished episodes will not return samples from the current episode')

        if sample_type == "random":
            indices = np.random.randint(low=0, high=buffersize, size=n)
        elif sample_type == 'last_n':
            indices = np.arange(buffersize - n, buffersize).astype(int)
        elif sample_type == 'prioritized':
            td_errors = np.abs(self.buffer['td'])
            p = td_errors / np.sum(td_errors)
            p += 0.00001  # lower bound p to ensure all states get sampled eventually
            p = td_errors / np.sum(td_errors)
            indices = np.random.choice(np.arange(buffersize), size=n, p=p)
        if return_indices:
            return self.buffer[indices], indices
        else:
            return self.buffer[indices]

    def sample_entire_episode(self):
        episode_idx = None
        while episode_idx is None:
            episode_idx = np.random.randint(len(self.episode_buffers))
        return self.episode_buffers[episode_idx]

    def sample_unique_positions(self, n):
        return self.unique_positions[np.random.randint(len(self.unique_positions), size=n)]

    def update_td_loss(self, indices, td_loss):
        self.buffer['td'][indices] = td_loss

    def load(self):
        logger.info('Attempting to load experience from disk')
        try:
            with open(self.accepted_episodes_path, 'r') as f:
                self.accepted_episodes_indices = ast.literal_eval(f.read())
        except Exception:
            logger.info('No experience could be loaded')
            return

        n_loaded_episodes = 0
        for i in self.accepted_episodes_indices[::-1]:
            if i is not None:
                buffer_path = os.path.join(self.path, 'episode_' + str(i).zfill(6), 'buffer.npz')
                with open(buffer_path, "rb") as f:
                    self.episode_buffers.insert(0, np.load(f, allow_pickle=True))
                n_loaded_episodes += 1
            if n_loaded_episodes >= self.keep_n_episodes:
                break

        self.combine_episode_buffers()
        logger.info('Experience loading successful')

    def save_episode(self, episode_idx=None):
        """save_episode

        :param episode_idx: int
            The index of the episode to be written to disk. If None (default), saves the last
            episode
        """
        # if we aren't told which episode to save to disk, save the last one
        if episode_idx is None:
            episode_idx =  len(self.episode_buffers) - 1
        self._write_episode_to_disk(episode_idx)
        self._write_accepted_episodes_indices_to_disk()

    def _write_episode_to_disk(self, episode_idx):
        """_write_episode_to_disk

        :param episode_idx: int
        """
        logger.debug('Writing episode to disk')
        episode_path = os.path.join(self.path, 'episode_{}'.format(str(episode_idx).zfill(6)))
        if not os.path.exists(episode_path):
            os.makedirs(episode_path)

        buffer_path = os.path.join(episode_path, 'buffer.npz')
        with open(buffer_path, "wb") as f:
            np.save(f, self.episode_buffers[episode_idx])

    def _write_accepted_episodes_indices_to_disk(self):
        logger.debug('Writing accepted episodes indices to disk')
        with open(self.accepted_episodes_path, 'w') as f:
            f.write(str(self.accepted_episodes_indices))
