from torchvision import datasets, transforms


def get_datasets(data_dir="./data"):
    transform = transforms.Compose(
        [  # chaining via Compose
            transforms.Pad(2),  # pad all borders by 2 → 28→32
            transforms.ToTensor(),  # PIL [0-255] -> FloatTensor [0.0-1.0]
            transforms.Normalize(
                (0.1307,), (0.3081,)
            ),  # (x–mean)/std using MNIST stats
        ]
    )

    train_data = datasets.MNIST(
        root=data_dir, train=True, download=True, transform=transform
    )
    test_data = datasets.MNIST(
        root=data_dir, train=False, download=True, transform=transform
    )
    return train_data, test_data
