import numpy as np
import torch 

def getangle(u,v):
    c = np.einsum("ij,ij->i", u, v) / np.linalg.norm(u, axis = 1) / np.linalg.norm(v, axis = 1)
    return np.rad2deg(np.arccos(np.clip(c, -1, 1)))

def clipangle(x):
    return (x + 2 * np.pi) % (2 * np.pi)

def get_phase(input_data):
    trajectory_data = input_data[276:432].reshape(13, 12)
    sign = 2 * trajectory_data[:, 4] - 1

    gating_data = input_data[4683:].reshape(-1,2)
    phase_sin = sign * gating_data[:,0].reshape(13, 29)[:,0]
    phase_cos = sign * gating_data[:,1].reshape(13, 29)[:,0]
    phase = clipangle(np.arctan2(phase_sin, phase_cos))

    return phase

def gen_gating_data(phase, trajectory_data, goal_data):
    root_pos = np.zeros((13, 3))
    root_pos[:,0] = trajectory_data[:,0]
    root_pos[:,2] = trajectory_data[:,1]
    root_dir = np.zeros((13,3))
    root_dir[:,0] = trajectory_data[:,2]
    root_dir[:,2] = trajectory_data[:,3]

    goal_pos = goal_data[:,:3]
    goal_pos[:,1] = 0
    goal_dir = goal_data[:,3:6]
    goal_dir[:, 1] = 0

    dist = np.linalg.norm(goal_pos - root_pos, axis = 1)[:,None]
    angle = getangle(goal_dir, root_dir)[:, None]

    trajectory_action = 2 * trajectory_data[:,4:] - 1
    goal_action = 2 * goal_data[:,6:] - 1
    goal_action = goal_action.repeat(3,1)
    goal_action[:,1::3] *= dist
    goal_action[:,2::3] *= angle

    phase = phase.repeat(29)
    X_ = np.concatenate([trajectory_action, goal_action], 1).flatten()
    gating_data = np.stack([X_ * np.sin(phase), X_ * np.cos(phase)], 1).flatten()
    
    return gating_data

def get_sample(input_data, output_data):
    return [
        torch.from_numpy(input_data[:, 0:432]).cuda(),
        torch.from_numpy(input_data[:, 432:601]).cuda(),
        torch.from_numpy(input_data[:, 601:2635]).cuda(),
        torch.from_numpy(input_data[:, 2635:4683]).cuda(),
        torch.from_numpy(input_data[:, 4683:5437]).cuda(),
        torch.from_numpy(output_data).cuda()
    ]

def normalize(X, mean, std):
    return (X - mean) / (1e-5 + std)

def unnormalize(X, mean, std):
    return X * std + mean

def update_phase(phase, phase_update):
    phase_new = phase.copy()
    phase_new[:6] += 2 * np.pi * phase_update[0]
    
    phase_new[6:] = phase[6]
    phase_new[6:] += 2 * np.pi * phase_update

    return clipangle(phase_new)