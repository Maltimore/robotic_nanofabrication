import numpy as np
np.set_printoptions(precision=3, floatmode='fixed', sign='+')
import logging
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
import os
from scipy.special import softmax

import util
import optimizer_util
import plotting


logger = logging.getLogger('general')
action_logger = logging.getLogger('action')
trajectory_logger = logging.getLogger('trajectory')
# just to append the episode beginning/end strings:
measurement_logger = logging.getLogger('measurement')


class Agent:
    def __init__(
            self, params, model, target_model, environment, episode, ruptures, successes,
            global_step, actions, reward, experience):
        '''
        :param params: dict
        :param model: pytorch model
        :param target)model: pytorch model
        :param environment: simulation environment
        :param episode: int
        :param ruptures: numpy array
            containing all previously occurred ruptures
        :param ruptures: numpy array
            containing all previously occurred successes (successful final lifting positions)
        :param global_step: int
            global training step
        :param actions: numpy array
            contains movements
        :param reward: reward object
            having the reward.calculate() function to compute the reward
        :param experience: Experience replay buffer
        '''
        self.params = params
        self.model = model
        self.target_model = target_model
        self.reward = reward
        self.actions = actions
        self.environment = environment
        self.experience = experience
        self.episode = episode if episode > 0 else -1
        self.global_step = global_step
        self.ruptures = ruptures
        self.successes = successes

        self.Q = []
        self.Value = []
        self.Advance = []
        self.rewards = []
        self.actions_hist = []
        self.episode_steps = 0
        self.trajectory = []
        self.last_target_update = 0

        self.optimizer = optimizer_util.get_torch_optimizer(model, params)

        # get angle from params and compute radians
        self.angle = (self.params['angle'] / 360) *  2 * np.pi
        # rotation matrix used to create grid points for plots
        self.rotation_matrix = np.array([[np.cos(self.angle), -np.sin(self.angle)],
                                         [np.sin(self.angle), np.cos(self.angle)]])

        # create actions which contain the original actions and also the
        # original actions going in opposite z direction
        self.actions_and_down_actions = np.concatenate((self.actions, self.actions), axis=0)
        self.actions_and_down_actions[int(self.actions_and_down_actions.shape[0] / 2):, 2] *= -1

        # value-plot grid parameters
        self.rmin = -17
        self.rmax =  17
        self.zmin =   0
        self.zmax =  15

        # set up plotting directories
        self.plots_path = os.path.join(params['output_path'], 'plots')
        self.trajectoryplots_basepath = os.path.join(self.plots_path, 'trajectory_plots')
        if not os.path.exists(self.trajectoryplots_basepath):
            os.makedirs(self.trajectoryplots_basepath)
        self.timeseriesplots_basepath = os.path.join(self.plots_path, 'timeseries_plots')
        if not os.path.exists(self.timeseriesplots_basepath):
            os.makedirs(self.timeseriesplots_basepath)

    def choose_action(self, q, T=0.):
        q = np.copy(q)
        p = softmax(q / T)
        # lower bound p
        p += 1e-6
        p /= np.sum(p)
        return np.random.choice(np.arange(q.size), p=p.ravel())

    def reset(self):
        logger.debug("Resetting worker")

        self.model.reset()
        self.target_model.reset()

        self.Q = []
        self.Value = []
        self.Advance = []
        self.rewards = []
        self.actions_hist = []
        self.position = np.zeros((3,), dtype=np.float32)
        self.episode_steps = 0
        self.episode_terminated = False
        self.episode_status = None

    def _check_end_of_episode(self, measurement):
        # measurement is a string like "success" or "failure"
        # if it is the end of the episode, otherwise it is
        # current I and force gradient df
        if isinstance(measurement, str):
            self.episode_terminated = True
            return True
        else:
            return False

    def initialize_state(self, measurement):
        self.trajectory = [self.position]

        # if it is already the end of the episode
        # (experimenter gave the interrupt signal),
        # do not try to initialize the state from measurements
        if self._check_end_of_episode(measurement):
            return

        # add the state for the current time
        current, df = measurement
        self.states = [np.hstack((np.zeros((3,), dtype=np.float32),
                                  np.array(current), np.array(df)))]
        return measurement

    def extract_features(self, state):
        """extract_features

        :param state: shape (n_batch, n_state_features)
        """
        return state[:, :self.params['n_features']]

    def perform_step(self):
        """
        walks one step in the environment
        """
        logger.debug("New step")

        # extract features after adding batch dimension
        features = self.extract_features(np.array(self.states[-1])[None, ...])
        #  predict action values
        q, value, advance = self.model.predict(features)
        logger.debug('Q-Values' + str(q))
        logger.debug('V-Values' + str(value))
        logger.debug('Advance' + str(advance))

        action_idx = self.choose_action(advance[0], self.params['action_temperature'])
        logger.debug('Chosen action (idx): {}'.format(action_idx))
        action_logger.info(action_idx)
        action = self.actions[action_idx]

        # update history
        self.actions_hist.append(action_idx)
        self.Q.append(q)
        self.Value.append(value)
        self.Advance.append(advance)

        # perform action
        old_position = self.position
        self.position = self.position + action
        logger.debug("New position: {}".format(self.position))
        self.environment.send_action(action)

        # receive measurement
        measurement = self.environment.receive_measurement()
        if self._check_end_of_episode(measurement):
            # set to unrealistic values cause they should never be used anyway
            current, df = 99999.0, 99999.0
        else:
            current, df = measurement

        # handle measurement
        if measurement == 'fail':
            self.episode_status = 'fail'
            self.ruptures = np.vstack((self.ruptures, self.position))
        elif measurement == 'success':
            self.episode_status = 'success'
            self.successes = np.vstack((self.successes, self.position))
        elif measurement == 'delete_last_episode':
            self.episode_status = 'delete_last_episode'

        # calculate reward
        reward = self.reward.calculate(old_position, self.position, self.ruptures, self.successes)[0]
        self.rewards.append(reward)

        # update states
        self.states.append(np.hstack((self.position, current, df)).astype(np.float32))

        # append next state to experience buffer, td error is set to high value such that
        # it will get sampled in experience replay
        self.experience.add([self.states[-2],
                             action_idx,
                             reward,
                             self.states[-1],
                             9999.0,
                             self.episode_terminated])

        # update trajectory
        self.trajectory.append(self.position)

        # log trajectory
        trajectory_logger.info(self.states[-1])

        return measurement

    def run_episode(self):
        self.episode += 1

        measurement = self.environment.receive_measurement()
        self.initialize_state(measurement)

        # logging
        log_string = 'START of episode {} '.format(self.episode)
        logger.info(log_string)
        action_logger.info(log_string)
        trajectory_logger.info(log_string)
        measurement_logger.info(log_string)

        # run episode
        while not self.episode_terminated:
            # perform step
            measurement = self.perform_step()
            self.episode_steps += 1
        self.trajectory = np.array(self.trajectory)  # was a list before

        # logging
        log_string = 'END of episode {} with measurement {} '.format(
                      self.episode, measurement) + \
                     'after {} steps'.format(self.episode_steps)
        logger.info(log_string)
        action_logger.info(log_string)
        trajectory_logger.info(log_string)
        measurement_logger.info(log_string)

    def get_episode_records(self):
        if self.episode_steps > 0:
            Q = np.vstack(self.Q)
            V = np.vstack(self.Value)
            A = np.vstack(self.Advance)
            R = np.array(self.rewards)
            actions = np.array(self.actions_hist)
            return Q, V, A, R, actions
        else:
            return [np.array([]) for i in range(5)]

    def train(self):
        # set number of train steps
        # in the beginning we shouldn't train too much
        train_steps = np.min([(self.episode + 1) * self.params['min_train_steps_per_episode'],
                              self.params['max_train_steps_per_episode']])

        for i in range(train_steps):
            # train step
            self.train_step()
            # update target network
            if self.global_step - self.last_target_update > self.params['target_update_rate']:
                #logger.info('Updating target model weights')
                self.target_model.load_state_dict(self.model.state_dict())
                self.last_target_update = self.global_step

    def model_based_sampling(self, n):
        orig_positions = self.experience.sample_unique_positions(n)
        positions = np.copy(orig_positions)
        # make random steps
        for i in range(positions.shape[0]):
            n_steps = np.random.randint(5)  # careful, randint is exclusive at the upper bound
            for _ in range(n_steps):
                new_pos = positions[i] \
                    + self.actions_and_down_actions[np.random.randint(len(self.actions_and_down_actions))]
                # only make step if it doesn't lead through a rupture point, such that we don't
                # have an s_t starting at a rupture
                # (final state action state tuple could still end in rupture, which is intended)
                if not np.isclose(new_pos[None, :], self.ruptures).all(axis=1).any():
                    positions[i] = new_pos
        s1 = positions
        a1 = np.random.randint(low=0, high=len(self.actions), size=s1.shape[0])
        s2 = s1 + self.actions[a1]
        r1 = self.reward.calculate(s1, s2, self.ruptures, self.successes)
        t2 = np.isclose(s2[:, None, :], self.ruptures).all(axis=2).any(axis=1)
        return s1, a1, r1, s2, t2

    def train_step(self):
        n_real_samples = round(self.params['batch_size'] * (1 - self.params['model_based_sampling_ratio']))
        n_model_samples = round(self.params['batch_size'] * self.params['model_based_sampling_ratio'])

        # these empty arrays is what we concatenate with if there are no real samples
        s1 = np.empty((0, 3), dtype=np.float)
        a1 = np.empty((0,), dtype=np.int)
        r1 = np.empty((0,), dtype=np.float)
        s2 = np.empty((0, 3), dtype=np.float)
        t2 = np.empty((0,), dtype=np.uint8)

        if n_real_samples > 0:
            # sample some states from experience
            sample, sample_indices = self.experience.sample(
                n_real_samples, sample_type='prioritized', return_indices=True)
            s1 = self.extract_features(np.stack(sample['s1']).reshape((n_real_samples, -1)))
            s2 = self.extract_features(np.stack(sample['s2']).reshape((n_real_samples, -1)))
            a1 = np.stack(sample['a1'])
            r1 = np.stack(sample['r1'])
            t2 = np.stack(sample['t2'])

        if n_model_samples > 0:
            # sample from model
            modeled_s1, modeled_a1, modeled_r1, \
                    modeled_s2, modeled_t2 = self.model_based_sampling(n_model_samples)
            s1 = np.concatenate((s1, modeled_s1), axis=0)
            a1 = np.concatenate((a1, modeled_a1), axis=0)
            r1 = np.concatenate((r1, modeled_r1), axis=0)
            s2 = np.concatenate((s2, modeled_s2), axis=0)
            t2 = np.concatenate((t2, modeled_t2), axis=0)

        # extract to torch
        s1 = torch.from_numpy(s1).float()
        s2 = torch.from_numpy(s2).float()
        r1 = torch.from_numpy(r1).float()
        t2 = torch.from_numpy(t2.astype('uint8'))

        # q2 for policy
        q2_livemodel, _, _ = self.model(s2)
        q2_livemodel.detach_()
        # q2 as determined by the target network
        q2_targetmodel, _, _ = self.target_model(s2)
        q2_targetmodel.detach_()
        # q1
        q1, _, _ = self.model(s1)
        q1 = q1[np.arange(q1.shape[0]), a1]

        # expected q2, using the value of q2 from the target network and
        # the probabilities of selecting the actions from the live network
        pi = torch.nn.functional.softmax(
                q2_livemodel / torch.Tensor([float(self.params['train_temperature'])]), dim=-1)  # Boltzmann
        target_q2 = torch.sum(pi * q2_targetmodel, dim=-1)

        # if it's the last step of an episode, set the q value of the next step to 0,
        # otherwise apply discount factor gamma
        q_tgt = r1 + torch.where(t2, torch.zeros_like(target_q2),
                                 self.params['gamma'] * target_q2)

        # criterion and train
        criterion = torch.nn.MSELoss()
        loss = criterion(q1, q_tgt)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # update the temporal difference loss in the experience buffer
        if n_real_samples > 0:
            td_loss = q_tgt - q1.detach()
            self.experience.update_td_loss(sample_indices, td_loss.numpy()[:n_real_samples])

        self.global_step += 1
        return loss

    def end_of_episode_plots(self):
        # trajectory plot
        step = 0.15
        grid = np.mgrid[self.rmin:self.rmax + step:step, self.zmin:self.zmax:step]
        grid = np.vstack((grid[[0]], np.zeros(grid[[0]].shape), grid[[1]]))
        gridshape = grid.shape
        grid = grid.reshape((3, -1)).T
        grid[:, :2] = (self.rotation_matrix @ grid[:, :2].T).T
        grid = grid.reshape((-1, 3)).astype(np.float32)
        q, v, a = self.model.predict(grid)
        v = v.reshape(gridshape[1:]).T
        v = np.flipud(v)

        # transform ruptures / successes coordinates
        ruptures_projected = util.project_into_plane(self.ruptures[:, :2], self.angle)
        ruptures_radius = np.linalg.norm(ruptures_projected, axis=1)
        ruptures_lateral_displacement = ruptures_radius * np.sign(ruptures_projected[:, 0])
        successes_projected = util.project_into_plane(self.successes[:, :2], self.angle)
        successes_radius = np.linalg.norm(successes_projected, axis=1)
        successes_lateral_displacement = successes_radius * np.sign(successes_projected[:, 0])

        fig = plt.figure(figsize=(16, 4))
        ax = fig.add_subplot(1, 3, 1)
        ax.imshow(v, extent=[self.rmin, self.rmax, self.zmin, self.zmax], vmin=-2.0, vmax=0.0, cmap='bwr')
        ax.scatter(ruptures_lateral_displacement, self.ruptures[:, 2], c='g', marker='x', s=0.9)
        ax.scatter(successes_lateral_displacement, self.successes[:, 2], c='g', marker='o', s=20.0)

        trajectory_projected = util.project_into_plane(self.trajectory[:, :2], self.angle)
        trajectory_radius = np.linalg.norm(trajectory_projected, axis=1)
        trajectory_lateral_displacement = trajectory_radius * np.sign(trajectory_projected[:, 0])
        ax.plot(trajectory_lateral_displacement, self.trajectory[:, 2], 'black')
        ax.set_xlabel('d [Å]')
        ax.set_ylabel('z [Å]')
        ax.set_xlim([self.rmin, self.rmax])
        ax.set_ylim([self.zmin, self.zmax])

        # 3d plot
        ax = fig.add_subplot(1, 3, 2, projection='3d')
        ax.plot(self.trajectory[:, 0], self.trajectory[:, 1], self.trajectory[:, 2])
        three_D_rmin = np.min(self.experience.unique_positions[:, :2])
        three_D_rmax = np.max(self.experience.unique_positions[:, :2])
        three_D_zmin = 0
        three_D_zmax = np.max(self.experience.unique_positions[:, 2])
        ax.set_xlim([three_D_rmin, three_D_rmax])
        ax.set_ylim([three_D_rmin, three_D_rmax])
        ax.set_zlim([three_D_zmin, three_D_zmax])

        # view from top
        ax = fig.add_subplot(1, 3, 3)
        ax.scatter(self.ruptures[:-1, 0], self.ruptures[:-1, 1], c='g', marker='x', s=5.0)
        ax.scatter(self.ruptures[[-1], 0], self.ruptures[[-1], 1], c='g', marker='x', s=10.0)
        ax.scatter(self.successes[:, 0], self.successes[:, 1], c='g', marker='o', s=20.0)
        ax.plot(self.trajectory[:, 0], self.trajectory[:, 1], linewidth=0.5)
        ax.set_xlim([self.rmin, self.rmax])
        ax.set_ylim([self.rmin, self.rmax])

        plot_path = os.path.join(self.trajectoryplots_basepath, str(self.episode).zfill(4) + '.png')
        plt.savefig(plot_path, dpi=300)
        plt.close()

        # timeseris plot
        tmp_path = os.path.join(self.timeseriesplots_basepath,
                                str(self.episode).zfill(3) + '_timeseries.png')
        Q, V, A, R, actions = self.get_episode_records()
        plotting.plot_episode(Q, V, A, R, actions, tmp_path)
