import argparse
import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import numpy as np
from tqdm.auto import tqdm
from collections import OrderedDict
import sys
import glob

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main(mode='vit_small'):
    metric_0 = []
    metric_1 = []
    metric_2 = []
    metric_3 = []
    epoch_list = list(range(20)) + list(range(20, 380, 5))
    for epoch in tqdm(epoch_list):
        activations = np.load(f'numpy_activations/{mode}/act_{epoch}.npy')
        targets = np.load(f'numpy_activations/{mode}/targ_{epoch}.npy')

        smaller_activations = activations.copy()

        mu_G = np.mean(smaller_activations, axis=0)

        mu_c = []
        for i in range(10):
            mu_c.append(np.mean(smaller_activations[targets == i], axis=0))

        avg_list = []
        for i in range(10):
            mu_cbar = mu_c[i] - mu_G
            avg_list.append(np.outer(mu_cbar, mu_cbar))
        sigma_B = np.mean(np.stack(avg_list), axis=0)

        avg_list = []
        for i in range(10):
            observations = smaller_activations[targets == i]
            class_mean = mu_c[i]
            
            outer_products = 0
            for i in range(5):
                mini_obs = observations[(i*observations.shape[0]//5):((i+1)*observations.shape[0]//5), :]
                outer_products += np.sum(np.einsum('ij,ik->ijk', (mini_obs - class_mean), (mini_obs - class_mean)), axis=0)
            avg_list.append(outer_products)
        sigma_W = np.sum(np.stack(avg_list), axis=0) / activations.shape[0]

        metric_0.append(np.trace(sigma_W @ np.linalg.pinv(sigma_B)) / 10)
        metric_1.append(np.trace(sigma_W)/np.trace(sigma_B))
        metric_2.append(np.trace(sigma_W))
        metric_3.append(np.trace(sigma_B))
    metric_0, metric_1, metric_2, metric_3 = np.array(metric_0), np.array(metric_1), np.array(metric_2), np.array(metric_3)
    np.save(f'metric_lists/{mode}/m0.npy', metric_0)
    np.save(f'metric_lists/{mode}/m1.npy', metric_1)
    np.save(f'metric_lists/{mode}/m2.npy', metric_2)
    np.save(f'metric_lists/{mode}/m3.npy', metric_3)

if(__name__ == '__main__'):
    parser = argparse.ArgumentParser(description="Run the model with specified mode.")
    parser.add_argument('mode', type=str, choices=['resnet18', 'vit_small'],
                        help='Model mode to run (either "resnet18" or "vit_small").')
    args = parser.parse_args()

    main(mode=args.mode)