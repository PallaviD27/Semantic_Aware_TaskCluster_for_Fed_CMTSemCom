import numpy as np
from numpy import where

# Import and prepare dataset
import torch
import torchvision
from torchvision import datasets, transforms
import options

def custom_skewed_partition(dataset, num_users, seed=None):

    assert num_users == 3

    # Target_Allocation = {
    #     0: [1975, 1974, 1974],
    #     1: [2190, 2552, 2000],
    #     2: [2340, 2079, 1539],
    #     3: [2000, 2131, 2000],
    #     4: [1947, 1947, 1948],
    #     5: [1807, 1807, 1807],
    #     6: [1718, 1500, 2700],
    #     7: [2089, 2088, 2088],
    #     8: [1951, 1942, 1958],
    #     9: [1983, 1980, 1986],
    # }
    # Iteration 1
    # Target_Allocation = {
    #     0: [1975, 0, 0],
    #     1: [0, 2552, 200],
    #     2: [2340, 2079, 0],
    #     3: [2000, 0, 500],
    #     4: [1947, 0, 1947],
    #     5: [0, 1807, 0],
    #     6: [0, 0, 2700],
    #     7: [2089, 700, 0],
    #     8: [200, 0, 1958],
    #     9: [0, 1980, 1986],
    # }
    #  Iteration2/3/4
    # Target_Allocation = {
    #     0: [0, 0, 1974],
    #     1: [0, 2552, 0],
    #     2: [2340, 2079, 0],
    #     3: [0, 500, 0],
    #     4: [0, 0, 1947],
    #     5: [0, 0, 1807],
    #     6: [0, 0, 2700],
    #     7: [2089, 0, 0],
    #     8: [1951, 0, 500],
    #     9: [1983, 0, 200],
    # }
    Target_Allocation = {
        0: [0, 0, 1974],
        1: [0, 2552, 0],
        2: [1000, 2079, 0],
        3: [0, 500, 0],
        4: [0, 0, 200],
        5: [0, 0, 500],
        6: [0, 500, 2000],
        7: [1500, 0, 0],
        8: [1551, 0, 500],
        9: [1583, 0, 200],
    }

    #  Get labels of train_dataset (argument to custom_skewed_partition in the get_dataset)
    y = dataset.targets.numpy()

    # Get count of each digit (how many times a digit/label appears in MNIST)
    counts = np.bincount(y, minlength=10)

    print('MNIST train counts per digit (0...9):', counts.tolist())

    # Fixing random seed so that the subsequent shuffling is reproducible
    rng = np.random.default_rng(seed)

    # Get indices for digits as arrays
    indices_by_digit = {d: np.where(y==d)[0] for d in range(10)}

    # Initialize blank dictionary for users
    dict_users = {u: [] for u in range(num_users)}
    print (dict_users)

    for d in range(10):
        rng.shuffle(indices_by_digit[d])

    # Sanity check if more samples are allocated than existing for any digit
    row_counts = [len(indices_by_digit[d]) for d in range(10)]
    for d in range(10):
        allocated = sum(Target_Allocation[d])
        available = row_counts[d]
        assert allocated <= available, (f"Digit {d}: allocation {allocated} > available {available}")

    # compute how many samples each client will get (for info + later checks)
    col_sums = [sum(Target_Allocation[d][u] for d in range(10)) for u in range(num_users)]
    print("Planned samples per client:", col_sums)

    for d in range(10):
        start = 0
        for u in range(num_users):
            k = Target_Allocation[d][u]
            if k == 0:
                continue
            dict_users[u].extend(indices_by_digit[d][start:start+k].tolist())
            start += k

    for u in range(num_users):
        dict_users[u] = np.array(dict_users[u], dtype=np.int64)
        assert len(dict_users[u]) == col_sums[u], (
            f"Client {u}: got {len(dict_users[u])} indices, expected {col_sums[u]}"
        )

    all_ids = np.concatenate([dict_users[u] for u in range(num_users)])
    assert len(all_ids) == len(set(all_ids.tolist())), "Overlap found between client splits"

    return dict_users


from collections import Counter


def print_label_distribution(dict_users, dataset):
    for client_id, indices in dict_users.items():
        labels = dataset.targets[indices]
        counts = Counter(labels.numpy())
        total = len(indices)
        print(f"\n Client {client_id} label distribution (Total:{total}):")
        for d in range(10):
            print(f" {d}: {counts.get(d, 0)}")




def get_dataset(args):

    if args.dataset == 'mnist':
        apply_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,),(0.3081,))])
        data_dir = './data/mnist/'
        train_dataset = datasets.MNIST(data_dir,train=True,download=True, transform=apply_transforms)
        test_dataset = datasets.MNIST(data_dir, train=False, download= True, transform=apply_transforms)

        # Sanity check for data size
        print(len(train_dataset))
        print(len(test_dataset))

        img,label = train_dataset[0]

        print(img.shape,label)

    #     If model selected is MLP (only option implemented) then do the label mapping as specified
        if args.model == 'MLP':
            label_mappings = {}
            def map_client_0(y):
                return 1 if y==2 else 0

            def map_client_1(y):
                if y == 1:
                    return 1
                elif y == 2:
                    return 2
                else:
                    return 0

            def map_client_2(y):
                return 1 if y == 6 else 0

            label_mappings = {
                0: map_client_0, # Function is just being referenced and not used
                1: map_client_1,
                2: map_client_2

            }

            for client_id, map_func in label_mappings.items():
                for y in range(10):
                    print(f'Client: {client_id}, Label: {y} -> Mapped to: {map_func(y)}')

        else:
            print('Invalid model selected')

        if args.partition_type == 'custom_skewed':
            user_groups = custom_skewed_partition(train_dataset, args.num_users, seed=args.seed)
            test_user_groups = {u: np.arange(len(test_dataset)) for u in range(args.num_users)}

        else:
            print('Invalid partition type')


    else:
        print('Invalid dataset selection')

    return train_dataset, test_dataset, user_groups, test_user_groups, label_mappings


def exp_details(args):
    print('\nExperimental Details:')
    print(f'    Model   : {args.model}')
    print(f'    Optimizer   : {args.optimizer}')
    print(f'    Learning    : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')
    print('     Federated parameters:')
    # if args.iid:
    #     print('     IID')
    #
    # else:
    #     print('     Non-IID')

    print(f'     Partition Type     : {args.partition_type}')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch Size    : {args.local_bs}')
    print(f'    Local Epochs    : {args.local_ep}\n')

    return
