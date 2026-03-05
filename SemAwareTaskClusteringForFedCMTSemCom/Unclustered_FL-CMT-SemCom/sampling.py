"""Start"""

import numpy as np


def custom_skewed_partition(dataset, num_users, seed=None):

 assert num_users == 3
# Iteration1
#  Target_Allocation = {
#   0: [1975, 0, 0],
#   1: [0, 2552, 200],
#   2: [2340, 2079, 0],
#   3: [2000, 0, 500],
#   4: [1947, 0, 1947],
#   5: [0, 1807, 0],
#   6: [0, 0, 2700],
#   7: [2089, 700, 0],
#   8: [200, 0, 1958],
#   9: [0, 1980, 1986],
#  }
#  Iteration2/3/4
#  Target_Allocation = {
#   0: [0, 0, 1974],
#   1: [0, 2552, 0],
#   2: [2340, 2079, 0],
#   3: [0, 500, 0],
#   4: [0, 0, 1947],
#   5: [0, 0, 1807],
#   6: [0, 0, 2700],
#   7: [2089, 0, 0],
#   8: [1951, 0, 500],
#   9: [1983, 0, 200],
#  }
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
 # Old allocation
 # Target_Allocation = {
 #  0: [1975, 1974, 1974],
 #  1: [2190, 2552, 2000],
 #  2: [2340, 2079, 1539],
 #  3: [2000, 2131, 2000],
 #  4: [1947, 1947, 1948],
 #  5: [1807, 1807, 1807],
 #  6: [1718, 1500, 2700],
 #  7: [2089, 2088, 2088],
 #  8: [1951, 1942, 1958],
 #  9: [1983, 1980, 1986],
 # }
 #  Get labels as a NumPy array; e.g. [5,0,9,1,...]
 y = dataset.targets.numpy()

 counts = np.bincount(y, minlength=10)
 print("MNIST train counts per digit (0...9:", counts.tolist())

 rng = np.random.default_rng(seed)
 indices_by_digit = {d: np.where(y==d)[0] for d in range(10)} # For each digit d (0 to 9) find all
                                                              # row indices whose label equals d
                                                              # gives result like {
                                                              #   0: array([  13,  102,  377, ...]), - positions where label==0
                                                              #   1: array([   5,   44,  210, ...]),
 # randomly shuffle indices per digit
 # If all “2” labels are at positions[7, 50, 88, 91, ...], after shuffling they might become[91, 7, 88, 50, ...].
 # Then,if Client 1 needs 2340 twos, we take the first 2340 from this shuffled list—i.e., a deterministic random sample of the “2” images.
 for d in range(10):
  rng.shuffle(indices_by_digit[d])

 # Sanity: row sums match dataset counts; column sums are 20k each
 # row_counts = [len(indices_by_digit[d]) for d in range(10)]
 # for d in range(10):
 #  assert sum(Target_Allocation[d]) == row_counts[d] , (
 #   f"Digit {d}: allocation { sum(Target_Allocation[d])} != dataset {row_counts[d]}"
 #  )
 # col_sums = [sum(Target_Allocation[d][u] for d in range(10)) for u in range(num_users)]
 # assert col_sums == [20000,20000,20000], f"Column sums must be [20000, 20000, 20000], got {col_sums}"

 # Sanity check if more samples are allocated than existing for any digit
 row_counts = [len(indices_by_digit[d]) for d in range(10)]
 for d in range(10):
  allocated = sum(Target_Allocation[d])
  available = row_counts[d]
  assert allocated <= available, (f"Digit {d}: allocation {allocated} > available {available}")

 # compute how many samples each client will get (for info + later checks)
 col_sums = [sum(Target_Allocation[d][u] for d in range(10)) for u in range(num_users)]
 print("Planned samples per client:", col_sums)

 # Slice per digit
 dict_users = {u: [] for u in range(num_users)}
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

def print_label_distribution(dict_users,dataset):
 for client_id, indices in dict_users.items():
  labels = dataset.targets[indices]
  counts = Counter(labels.numpy())
  total = len(indices)
  print(f"\n Client {client_id} label distribution (Total:{total}):")
  for d in range(10):
   print(f" {d}: {counts.get(d, 0)}")




# Second type of skewed dataset
# def custom_skewed_partition1(dataset, num_users, seed=None):
#  """
#  Custom non-IID and equal partition for clients:
#   - Client 0: 20% of digit 2, 80% from others (excluding 2)
#   - Client 1: 15% each of digits 1, 2, 3, and 55% from others (excluding 1,2,3)
#   - Client 2: 20% of digit 6, 80% from others (excluding 6)
#  Ensures: no overlap across clients; exact samples_per_client per client.
#  """
#  import numpy as np
#  rng = np.random.default_rng(seed)
#  targets = dataset.targets.numpy()
#  total_samples = len(targets)
#  samples_per_client = total_samples // num_users  # 60000 // 3 = 20000
#
#  # Pool of available indices per digit (mutable)
#  available_by_digit = {d: np.where(targets == d)[0].tolist() for d in range(10)}
#  for d in available_by_digit:
#   rng.shuffle(available_by_digit[d])
#
#  # Bias config per client
#  bias_config = {
#   0: {"include": {2: 0.10}, "exclude": [2]},
#   1: {"include": {1: 0.15, 2: 0.12, 3: 0.15}, "exclude": [1, 2, 3]},
#   2: {"include": {6: 0.20}, "exclude": [6]},
#  }
#
#  def take_from_digit(digit, k):
#   """Pop k indices from this digit’s pool."""
#   k = min(k, len(available_by_digit[digit]))
#   taken = available_by_digit[digit][:k]
#   del available_by_digit[digit][:k]
#   return taken
#
#  dict_users = {}
#
#  for client_id in range(num_users):
#   config = bias_config[client_id]
#   include = config["include"]
#   exclude = set(config["exclude"])
#
#   client_indices = []
#
#   # 1) Add biased samples
#   for digit, ratio in include.items():
#    need = int(round(ratio * samples_per_client))
#    client_indices.extend(take_from_digit(digit, need))
#
#   # 2) Fill the remainder from "others" digits (excluding `exclude`)
#   remaining_needed = samples_per_client - len(client_indices)
#   if remaining_needed > 0:
#    # Build a unified pool of (digit, count_available) for allowed digits
#    allowed_digits = [d for d in range(10) if d not in exclude]
#    # Keep drawing until we hit the target or pools are empty
#    while remaining_needed > 0 and any(len(available_by_digit[d]) > 0 for d in allowed_digits):
#     # Pick a digit with available samples
#     d_choices = [d for d in allowed_digits if len(available_by_digit[d]) > 0]
#     d = rng.choice(d_choices)
#     take = min(remaining_needed, len(available_by_digit[d]), 512)  # draw in chunks for speed
#     client_indices.extend(take_from_digit(d, take))
#     remaining_needed -= take
#
#   # Final safety check: if we couldn’t fill (ran out of data), trim to exact length
#   client_indices = client_indices[:samples_per_client]
#   dict_users[client_id] = np.array(client_indices, dtype=np.int64)
#
#  # Optional sanity checks
#  # - no overlap across clients
#  all_ids = np.concatenate([dict_users[i] for i in range(num_users)])
#  assert len(all_ids) == len(set(all_ids.tolist())), "Overlap found between client splits!"
#  # - exactly equal sizes
#  for i in range(num_users):
#   assert len(dict_users[i]) == samples_per_client, f"Client {i} size mismatch."
#
#  return dict_users
#
#
# from collections import Counter
# import numpy as np
#
# def print_label_distribution(dict_users, dataset):
#     for client_id, indices in dict_users.items():
#         labels = dataset.targets[indices]
#         label_counts = Counter(labels.numpy())
#         total_samples = len(indices)
#         print(f"\nClient {client_id} label distribution (Total:{total_samples} samples):")
#         for label in sorted(label_counts):
#             print(f"  Label {label}: {label_counts[label]} samples")
#
#
# # mnist_train = datasets.MNIST('./data', train= True, download=True, transform=transforms.ToTensor())
# # dict_users = custom_skewed_partition(mnist_train, num_users=3)
# # print_label_distribution(dict_users, mnist_train)