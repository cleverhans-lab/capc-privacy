from architecture.mnist_net import MnistNet
import torch
import os


def get_model(dataset: str, checkpoint_dir: str, device):
    model_file = os.path.join(checkpoint_dir, dataset, dataset + '.pt')
    if dataset == 'mnist':
        model = MnistNet().to(device)
        model.load_state_dict(torch.load(model_file))

    return model
