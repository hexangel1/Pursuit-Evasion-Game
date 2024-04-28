#!/usr/bin/env python3

import argparse
from copy import deepcopy
from ddpg import DDPG
from utils import *


def make_plots(metrics_dict, validate_iter, save_path):
    append_metric_to_json("output/metrics.json", metrics_dict)
    plot_rewards(metrics_dict['reward'], validate_iter, save_path)
    plot_distances(metrics_dict['min'], metrics_dict['avg'], validate_iter, save_path)
    prGreen('Plots saved to {}'.format(save_path))


def validate_model(ddpg_net, num_episodes):

    ddpg_net.eval()
    policy = lambda x: ddpg_net.select_action(x, decay_epsilon=False)

    curr_state = None
    played_episodes = 0
    episode_rewards = []
    episode_distances_min = []
    episode_distances_avg = []
    episode_reward = 0.

    prGreen("VALIDATION STARTED")
    while played_episodes < num_episodes:
        if curr_state is None:
            curr_state, _, _, _, _ = get_state_reward()
            ddpg_net.reset(curr_state)

        next_state, reward, done, distance, _ = env_step(policy(curr_state))
        curr_state = deepcopy(next_state)
        episode_reward += reward

        if done:
            curr_state = None
            episode_rewards.append(episode_reward)
            episode_distances_min.append(distance[0])
            episode_distances_avg.append(distance[1])
            played_episodes += 1
            prGreen("{}/{}: reward {}, distance {}".format(played_episodes, num_episodes, episode_reward, distance))
            episode_reward = 0.

    ddpg_net.train()
    prGreen("VALIDATION FINISHED")
    return sum(episode_rewards) / len(episode_rewards),\
           sum(episode_distances_min) / len(episode_distances_min),\
           sum(episode_distances_avg) / len(episode_distances_avg)


def train(num_iterations, ddpg_net, resume_path, save_path, save_model_iter=1000, validate_model_iter=0, debug=False):

    if resume_path:
        ddpg_net.load_weights(resume_path)
    ddpg_net.train()
    step = episode = 0
    episode_reward = 0.
    metrics_dict = {"reward": [], "min": [], "avg": []}
    curr_state = None
    advice = None
    need_validation = (validate_model_iter > 0)

    while step < num_iterations:
        if step % 100 == 0:
            print(step, "/", num_iterations)
        # reset if it is the start of episode
        if curr_state is None:
            curr_state, _, _, _, advice = get_state_reward()
            ddpg_net.reset(curr_state)

        # agent select action
        action = ddpg_net.select_action(curr_state, advice=advice, use_advice=(episode == 0))

        # env response with next_state, reward, terminate flag
        next_state, reward, done, _, advice = env_step(action)

        # agent observe and update policy
        ddpg_net.observe(reward, next_state, done)
        if not need_validation and episode > 0:
            ddpg_net.update_policy()
            step += 1
            if step % save_model_iter == 0:
                ddpg_net.save_model(save_path)

        if validate_model_iter > 0 and step % validate_model_iter == 0:
            need_validation = True

        episode_reward += reward
        curr_state = deepcopy(next_state)

        if done: # end of episode
            if debug:
                prGreen('#{}: episode_reward:{} steps:{}'.format(episode, episode_reward, step))

            ddpg_net.memory.append(
                curr_state,
                ddpg_net.select_action(curr_state),
                0., False
            )
            if need_validation:
                validation_reward, min_dist, avg_dist = validate_model(ddpg_net, 10)
                metrics_dict['reward'].append(validation_reward)
                metrics_dict['min'].append(min_dist)
                metrics_dict['avg'].append(avg_dist)
                need_validation = False
            curr_state = None
            episode_reward = 0.
            episode += 1

    if validate_model_iter > 0:
        make_plots(metrics_dict, validate_model_iter, save_path)


def test(num_iterations, ddpg_net, resume_path, debug=False):

    ddpg_net.load_weights(resume_path)
    ddpg_net.eval()
    policy = lambda x: ddpg_net.select_action(x, decay_epsilon=False)

    curr_state = None
    for i in range(num_iterations):
        if curr_state is None:
            curr_state, _, _, _, _ = get_state_reward()
            ddpg_net.reset(curr_state)

        next_state, reward, done, _, _ = env_step(policy(curr_state))
        curr_state = deepcopy(next_state)
        if done:
            curr_state = None


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='DDPG Network')

    parser.add_argument('--mode', default='train', type=str, help='support option: train/test')
    parser.add_argument('--hidden1', default=40, type=int, help='hidden num of first fully connect layer')
    parser.add_argument('--hidden2', default=30, type=int, help='hidden num of second fully connect layer')
    parser.add_argument('--rate', default=0.001, type=float, help='learning rate')
    parser.add_argument('--prate', default=0.0001, type=float, help='policy net learning rate (only for DDPG)')
    parser.add_argument('--warmup', default=1000, type=int, help='time without training but only filling the replay memory')
    parser.add_argument('--discount', default=0.99, type=float, help='discount')
    parser.add_argument('--bsize', default=64, type=int, help='minibatch size')
    parser.add_argument('--rmsize', default=6000000, type=int, help='memory size')
    parser.add_argument('--window_length', default=2, type=int, help='window length')
    parser.add_argument('--tau', default=0.001, type=float, help='moving average for target network')
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--init_w', default=0.003, type=float, help='') 
    parser.add_argument('--train_iter', default=151000, type=int, help='max train iterations')
    parser.add_argument('--test_iter', default=20000, type=int, help='max test iterations')
    parser.add_argument('--validate_iter', default=2500, type=int, help='validate model each N iterations')
    parser.add_argument('--epsilon', default=50000, type=int, help='linear decay of exploration policy')
    parser.add_argument('--seed', default=42, type=int, help='random seed')
    parser.add_argument('--resume', default=0, type=int, help='resuming model path for testing')
    parser.add_argument('--train_resume', default=-1, type=int, help='resuming model path for training')

    args = parser.parse_args()

    set_random_seed(args.seed)
    env_connect()

    ddpg_net = DDPG(state_size, action_size, args)

    if args.mode == 'train':
        save_path = get_output_folder()
        resume_path = get_resume_folder(args.train_resume)
        train(args.train_iter, ddpg_net, resume_path, save_path,
              validate_model_iter=args.validate_iter,debug=args.debug)

    elif args.mode == 'test':
        resume_path = get_resume_folder(args.resume)
        test(args.test_iter, ddpg_net, resume_path, debug=args.debug)

    else:
        raise RuntimeError('undefined mode {}'.format(args.mode))
