import torch
import torchvision.transforms as transforms
from _cifar10 import cifar10_dataset
try:
    from ._cifar10 import cifar10_dataset
except Exception:
    import os, sys
    # project root = parent of the `kim` directory
    proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if proj_root not in sys.path:
        sys.path.insert(0, proj_root)
    try:
        # try importing the sibling module as a top-level module
        from _cifar10 import cifar10_dataset
    except Exception as e:
        raise ImportError("Could not import cifar10_dataset from ._cifar10 or _cifar10") from e
class cifar100_dataset(cifar10_dataset):   
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }
    cls_num = 100
if __name__ == '__main__':
    import pdb
    pdb.set_trace()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainset = cifar100_dataset(root='./data',
                                train=True,
                                download=True,
                                transform=transform)
    trainloader = iter(trainset)
    data, label = next(trainloader)
    