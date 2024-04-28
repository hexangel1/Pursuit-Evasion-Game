import os
import time
import json
import socket
import struct
import random
import torch
import numpy as np
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

client_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)

pursuers_amount = 2
evaders_amount = 1
histogram_size = 12
agent_state_size = 4
state_size = pursuers_amount * (4 + histogram_size) + evaders_amount * 4
action_size = pursuers_amount * 2
message_bytes = state_size * 8 + 8 + 4 + 2 * 8 + action_size * 8

action_format = str(action_size) + "d"
response_format = "=" + str(state_size) + "ddidd" + str(action_size) + "d"

def prRed(prt): print("\033[91m {}\033[00m" .format(prt))
def prGreen(prt): print("\033[92m {}\033[00m" .format(prt))

def to_numpy(tensor):
    return tensor.cpu().detach().numpy()

def to_tensor(ndarray):
    return torch.from_numpy(ndarray.astype(np.float32)).to(device)

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

def get_output_folder():
    work_dir = "output"
    os.makedirs(work_dir, exist_ok=True)
    experiment_id = 0
    for folder_name in os.listdir(work_dir):
        if not os.path.isdir(os.path.join(work_dir, folder_name)):
            continue
        try:
            folder_name = int(folder_name.split('-run')[-1])
            if folder_name > experiment_id:
                experiment_id = folder_name
        except:
            pass
    experiment_id += 1

    work_dir = os.path.join(work_dir, "pursuit") + '-run{}'.format(experiment_id)
    os.makedirs(work_dir, exist_ok=True)
    return work_dir

def get_resume_folder(experiment_id):
    work_dir = "output"
    if experiment_id == -1:
        return None
    if experiment_id == 0:
        return os.path.join(work_dir, "best_weights")
    return os.path.join(work_dir, "pursuit") + '-run{}'.format(experiment_id)

def set_random_seed(seed):
    if seed is None:
        seed = int(time.time())
        seed = ((seed & 0xff000000) >> 24) + ((seed & 0x00ff0000) >> 8) +\
               ((seed & 0x0000ff00) <<  8) + ((seed & 0x000000ff) << 24)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def gracefully_finish(message = "Gracefully finished"):
    client_socket.close()
    print(message)
    exit()

def env_connect():
    try:
        client_socket.connect('./Robot-Control/ddpg-bridge.sock')
    except socket.error:
        gracefully_finish(message="Connection failed")

def make_action(action):
    data = struct.pack(action_format, *action)
    try:
        client_socket.send(data)
    except socket.error:
        gracefully_finish()

def get_state_reward():
    try:
        data = client_socket.recv(message_bytes)
    except socket.error:
        gracefully_finish()
    if len(data) == 0:
        gracefully_finish()
    res = struct.unpack(response_format, data)
    return np.array(res[:state_size]), res[state_size], bool(res[state_size+1]),\
        (res[state_size+2], res[state_size+3]), np.array(res[state_size+4:])

def env_step(action):
    make_action(action)
    return get_state_reward()

def normalize_action(action):
    result = []
    for i in range(0, len(action), 2):
        x = action[i]
        y = action[i+1]
        norm = np.sqrt(x**2 + y**2)
        result.append(x / norm)
        result.append(y / norm)
    return np.array(result)


def append_metric_to_json(filename, new_metric_data):
    metrics = {}

    if os.path.isfile(filename):
        with open(filename, 'r') as jsonfile:
            metrics = json.load(jsonfile)

    metrics["N" + str(pursuers_amount)] = new_metric_data

    with open(filename, 'w') as jsonfile:
        json.dump(metrics, jsonfile)


def plot_rewards(rewards, validate_iter, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(range(0, len(rewards)*validate_iter, validate_iter),
             rewards, marker='o', color='b', label='average')

    plt.title('Total reward per episode')
    plt.xlabel('Training iterations')
    plt.ylabel('Total reward')
    plt.grid(True)
    plt.legend()
    plt.savefig('{}/reward_plot.png'.format(save_path))


def plot_distances(min_dist, avg_dist, validate_iter, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot((range(0, len(min_dist)*validate_iter, validate_iter)),
             min_dist, marker='o', color='r', label='group minimal')
    plt.plot((range(0, len(avg_dist)*validate_iter, validate_iter)),
             avg_dist, marker='o', color='b', label='group average')

    plt.title('Final distance from pursuers group to evader')
    plt.xlabel('Training iterations')
    plt.ylabel('Final distance')
    plt.grid(True)
    plt.legend()
    plt.savefig('{}/distance_plot.png'.format(save_path))
