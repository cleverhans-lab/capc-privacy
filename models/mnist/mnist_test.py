import torch

from models.mnist.mnist_train import set
from models.mnist.mnist_train import test


def main():
    result = set()
    model = result['model']
    args = result['args']
    device = result['device']
    train_loader = result['train_loader']
    test_loader = result['test_loader']
    model.load_state_dict(torch.load(args.saved_model_name))

    print('train set:')
    test(model=model, test_loader=train_loader, device=device)

    print('test set: ')
    test(model=model, test_loader=test_loader, device=device)


if __name__ == "__main__":
    main()
