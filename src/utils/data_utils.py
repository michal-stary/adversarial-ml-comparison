import torch
from torchvision import transforms, datasets


def create_loaders(data_dir, task_config, batch_size, transform=None, random_state=None,
                   n_workers=0, n_samples=1000):
    tr = [transforms.ToTensor()]
    if transform is not None:
        tr = [transforms.ToTensor(), transforms.Lambda(transform)]
    transform = transforms.Compose(tr)

    if task_config == 'CIFAR10':
        dataset_test = datasets.CIFAR10(data_dir, train=False, transform=transform, download=True)
    if task_config == 'MNIST':
        dataset_test = datasets.MNIST(data_dir, train=False, transform=transform, download=True)
    if task_config == 'IMAGENET':
        dataset_test = datasets.ImageNet(data_dir, train=False, transform=transform, download=True)

    if random_state is not None:
        torch.manual_seed(random_state)

    perm = torch.randperm(len(dataset_test))[:n_samples]
    shuffled = torch.utils.data.Subset(dataset_test, perm)
    loader_test = torch.utils.data.DataLoader(shuffled, batch_size=batch_size,
                                              shuffle=False, num_workers=n_workers)

    return loader_test
