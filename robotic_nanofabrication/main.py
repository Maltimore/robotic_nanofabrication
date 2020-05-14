import argparse
import logging
import os
import sys
import time
import tarfile
import shutil
import numpy as np
from ruamel.yaml import YAML
import torch

# imports from this project
import util
import environment
from reward import Reward
import agents
from experience import ExperienceBuffer
import models

yaml = YAML()


def main(params):
    start_time = time.time()
    output_path = params['output_path']
    if 'gridsearch' in params.keys():
        gridsearch_results_path = params['gridsearch_results_path']
        gridsearch = params['gridsearch']
    else:
        gridsearch = False

    plot_interval = 10 if gridsearch else 1

    with open(os.path.join(output_path, 'parameters.yaml'), 'w') as f:
        yaml.dump(params, f)

    # LOGGING
    all_loggers = []
    formatter = logging.Formatter('%(asctime)s:%(msecs)03d-%(levelname)s-%(message)s',
                                  datefmt='%H:%M:%S')
    # General logger
    logger = logging.getLogger('general')
    logger.setLevel(logging.DEBUG)
    general_level_file_handler = logging.FileHandler(os.path.join(output_path, 'general_log.txt'))
    general_level_file_handler.setLevel(logging.INFO)
    general_level_file_handler.setFormatter(formatter)
    logger.addHandler(general_level_file_handler)
    if not gridsearch:  # only save debug logs for real experiments (debug logs are huge)
        debug_level_file_handler = logging.FileHandler(os.path.join(output_path, 'debug_log.txt'))
        debug_level_file_handler.setLevel(logging.DEBUG)
        debug_level_file_handler.setFormatter(formatter)
        logger.addHandler(debug_level_file_handler)
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    all_loggers.append(logger)
    # Received-measurement logger
    measurement_logger = logging.getLogger("measurement")
    handler = logging.FileHandler(os.path.join(output_path, 'measurement_logs.txt'))
    handler.setFormatter(formatter)
    measurement_logger.addHandler(handler)
    measurement_logger.setLevel(logging.DEBUG)
    all_loggers.append(measurement_logger)
    # Action logger
    action_logger = logging.getLogger("action")
    handler = logging.FileHandler(os.path.join(output_path, 'action_logs.txt'))
    handler.setFormatter(formatter)
    action_logger.addHandler(handler)
    action_logger.setLevel(logging.DEBUG)
    all_loggers.append(action_logger)
    # Trajectory logger
    trajectory_logger = logging.getLogger("trajectory")
    handler = logging.FileHandler(os.path.join(output_path, 'trajectory_logs.txt'))
    handler.setFormatter(formatter)
    trajectory_logger.addHandler(handler)
    trajectory_logger.setLevel(logging.DEBUG)
    all_loggers.append(trajectory_logger)

    # create results directories
    model_dir = os.path.join(output_path, 'models')
    data_basepath = os.path.join(output_path, 'data')
    all_paths = [model_dir, data_basepath]
    for _path in all_paths:
        if not os.path.exists(_path):
            os.makedirs(_path)

    # initialize/load program state
    start_program_state = {
        'episode': -1,
        'global_step': 0,
        'first_success': None,
        'best_episode_score': 0,
        'best_episode_index': 0,
        'output_path': output_path,
        'git_head': util.get_git_revision_hash(),
        'run_finished': False,
        'start_time': time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(start_time))
    }
    program_state_path = os.path.join(output_path, 'program_state.yaml')
    try:
        with open(program_state_path, 'r') as f:
            program_state = yaml.load(f)
        logger.info('Loaded program_state')
    except Exception:
        logger.debug('Did not find program_state.dump file to restore program state, starting new')
        program_state = start_program_state

    # Experience buffer
    exppath = os.path.join(output_path, 'experience')
    if not os.path.exists(exppath):
        os.makedirs(exppath)
    expbuffer_dtype = [('s1', np.ndarray),
                       ('a1', np.int),
                       ('r1', np.float),
                       ('s2', np.ndarray),
                       ('td', np.float),
                       ('t2', np.bool)]
    exp = ExperienceBuffer(exppath, keep_n_episodes=params['keep_n_episodes'], buffer_dtype=expbuffer_dtype)
    if len(exp.episode_buffers) - 1 != program_state['episode']:
        error_msg = 'Episode index found in program_state does not match the episodes in ' + \
                    'the experience buffer. Did you delete experience? In this case, you can ' + \
                    'modify program_state.yaml accordingly. This involves setting the value ' + \
                    'for episode to -1. Currently, we would see inconsistent indices in the ' + \
                    'experience buffer and the log files. Should we continue anyways? (y/n)'
        if input(error_msg) != 'y':
            sys.exit()
    all_loggers.append(exp.logger)
    logger.info('Experience size: ' + str(len(exp)))

    # ACTIONS
    x_y_step = 0.3
    z_step = 0.1
    actions = np.array([
        [0., 0., z_step],  # up
        [0., -x_y_step, z_step],  # forward
        [0., x_y_step, z_step],  # backward
        [-x_y_step, 0., z_step],  # left
        [x_y_step, 0., z_step],   # right
    ], dtype=np.float32)
    n_actions = len(actions)

    # setup models
    # Reward
    reward = Reward(step_reward=params['step_reward'],
                    fail=params['fail_reward'], success=params['success_reward'])
    # QModels
    model_module = getattr(models, params['model'])
    model = model_module.Model(params, n_actions)
    target_model = model_module.Model(params, n_actions)
    current_model_path = os.path.join(output_path, 'models', 'current')
    previous_model_path = os.path.join(output_path, 'models', 'previous')
    try:
        logger.info('Loading model from: ' + current_model_path)
        model = torch.load(current_model_path)
        target_model = torch.load(current_model_path)
    except Exception:
        # no saved models found, saving randomly initiated ones
        model.save(current_model_path)
        model.save(previous_model_path)

    # environment
    if params['environment'] == 'toy':
        env = environment.ToyTrajectoryEnvironment(
            os.path.join("trajectories", params['train_trajectory_name']),
            lower_hose_radius=params['lower_hose_radius'],
            project_angle=np.pi / 4)
    elif params['environment'] == 'toy_sophisticated':
        env = environment.SophisticatedToyTrajectoryEnvironment(
            os.path.join("trajectories", params['train_trajectory_name']),
            lower_hose_radius=params['lower_hose_radius'],
            upper_hose_radius=params['upper_hose_radius'],
            project_angle=np.pi / 4)
    elif params['environment'] == 'microscope':  # if connected to the SPM
        env = environment.STMEnvironment(params['host'], params['port'])
    else:
        raise Exception('Unrecognized environment {} requested'.format(params['environment']))

    # ruptures and successes arrays
    rupt_file = os.path.join(output_path, 'ruptures.npy')
    if os.path.exists(rupt_file):
        ruptures = np.load(rupt_file)
    else:
        ruptures = np.empty((0, 3))
    succ_file = os.path.join(output_path, 'successes.npy')
    if os.path.exists(succ_file):
        successes = np.load(succ_file)
    else:
        successes = np.empty((0, 3))

    # agent
    agent_module = getattr(agents, params['agent'])
    agent = agent_module.Agent(
        params=params,
        model=model,
        target_model=target_model,
        environment=env,
        experience=exp,
        actions=actions,
        reward=reward,
        ruptures=ruptures,
        successes=successes,
        global_step=program_state['global_step'],
        episode=program_state['episode'])

    logger.info('Starting at training step ' + str(agent.global_step))
    best_reward = 0
    while True:
        agent.reset()
        env.reset()

        agent.run_episode()
        exp.finish_episode()
        Q, V, A, R, actions = agent.get_episode_records()
        episode_status = agent.episode_status

        if agent.episode % plot_interval == 0 and agent.episode_steps > 0:
            agent.end_of_episode_plots()
        if episode_status == 'delete_last_episode':
            # SPM experimenter wants to delete the last episode
            logger.info('Deleting last episode and reload model from one episode before')
            exp.delete_episode(agent.episode)
            # if we received delete_last_episode while having 0 steps in the current episode,
            # then the protocol is to actually to delete the previous episode.
            if agent.episode_steps == 0:
                logger.info('I received delete_last_episode while the current episode still ' +
                            'had not started, which means I should actually delete the previous ' +
                            'episode. I will do that now')
                exp.delete_episode(agent.episode - 1)
                # reload previous model
                logger.info('Load model from: ' + previous_model_path)
                model = torch.load(previous_model_path)
                target_model = torch.load(previous_model_path)
        elif episode_status in ['fail', 'success']:
            # Training
            agent.train()
            # Logging
            logger.info('Global training step: {}'.format(agent.global_step))
            # save best episode
            current_reward = np.sum(R)
            if current_reward > best_reward:
                best_reward = current_reward
                program_state['best_episode_score'] = agent.episode_steps
                program_state['best_episode_index'] = agent.episode
        # update program state
        program_state['episode'] = int(agent.episode)
        program_state['global_step'] = int(agent.global_step)
        if episode_status == 'success' and program_state['first_success'] is None:
            program_state['first_success'] = int(agent.episode)

        # save everything to disk. That process should not be interrupted.
        with util.Ignore_KeyboardInterrupt():
            data_path = os.path.join(data_basepath, str(agent.episode).zfill(3) + '_data')
            with open(program_state_path, 'w') as f:
                yaml.dump(program_state, f)
            # put program_state into gridsearch results folder
            if gridsearch:
                with open(os.path.join(gridsearch_results_path, 'program_state.yaml'), 'w') as f:
                    yaml.dump(program_state, f)
            exp.save_episode()
            np.savez(data_path, Q=Q, A=A, V=V, actions=actions, reward=R)
            # on disk, copy the current model file to previous model, then save the current
            # in-memory model to disk as the current model
            if not episode_status == 'delete_last_episode':
                shutil.copy(current_model_path, previous_model_path)
            # save the model parameters both in the file for the current model, as well as
            # in the file for the model for the current episode
            model.save(current_model_path)
            model.save(os.path.join(output_path, 'models', 'episode_{}'.format(agent.episode)))
            np.save(rupt_file, agent.ruptures)
            np.save(succ_file, agent.successes)

        # handle end of run (lifting success, or more than stop_after_episode episodes)
        if episode_status == 'success' or agent.episode >= params['stop_after_episode']:
            program_state['run_finished'] = True
            end_time = time.time()
            program_state['run_time'] = end_time - start_time
            program_state['end_time'] =  time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(end_time))
            with open(program_state_path, 'w') as f:
                yaml.dump(program_state, f)
            # put program_state into gridsearch results folder
            if gridsearch:
                with open(os.path.join(gridsearch_results_path, 'program_state.yaml'), 'w') as f:
                    yaml.dump(program_state, f)
                # if this is a gridsearch, pack the output files into a .tar.gz
                # and delete the output folder
                # close all logger filehandlers such that we can delete the folder later
                for logger in all_loggers:
                    for handler in logger.handlers[:]:
                        handler.close()
                        logger.removeHandler(handler)
                tarfile_path = output_path + '.tar.gz'
                with tarfile.open(tarfile_path, "w:gz") as tar:
                    tar.add(output_path, arcname=os.path.basename(output_path))
                    tar.close()
                # delete the output folder
                shutil.rmtree(output_path, ignore_errors=True)
                # end of "if gridsearch"
            print('Run finished at episode: {}'.format(agent.episode))
            return


if __name__ == '__main__':
    with open(os.path.join('robotic_nanofabrication', 'parameters.yaml'), 'r') as f:
        params = dict(yaml.load(f))
        params = params['default']

    # cmd line args
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', default='outfiles/test1', help='destination of results')
    parser.add_argument('--host', default='134.94.242.90', help='server ip')
    parser.add_argument('--port', default=5000, type=int, help='server port')
    args = parser.parse_args()
    params['output_path'] = args.output_path
    params['host'] = args.host
    params['port'] = args.port

    # create main output directory
    if not os.path.exists(params['output_path']):
        os.makedirs(params['output_path'])
    else:
        inpt = input('WARNING: Output directory already exists! ' +
                     'Continue training? [y/N] (default: y)')
        if inpt.capitalize() == 'N':
            sys.exit('Ok exiting')

    with open(os.path.join(params['output_path'], 'parameters.yaml'), 'w') as f:
        yaml.dump(params, f)

    try:
        main(params)
    except KeyboardInterrupt:
        sys.exit('KeyboardInterrupt')
    except Exception:
        import traceback
        ty, value, tb = sys.exc_info()
        traceback.print_exc()
        try:
            import pdb
            pdb.post_mortem(tb)
        except ImportError:
            import pdb
            pdb.post_mortem(tb)
