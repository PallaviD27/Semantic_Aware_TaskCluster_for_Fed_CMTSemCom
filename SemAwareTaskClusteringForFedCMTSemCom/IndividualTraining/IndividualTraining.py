# Three clients, skewed data, individual performance

import torch

# Individual_main.py  (Scenario 3: 3 independent clients, no aggregation)

import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm
import tikzplotlib

import torch
import sys
from tensorboardX import SummaryWriter

from options import args_parser
from update import test_inference, LocalUpdate
from model import Encoder_model, ClientModel
from sampling import get_dataset, exp_details
from collections import defaultdict
from torch.utils.data import Subset

# os.makedirs('././save', exist_ok=True)

# os.makedirs('./IndividualTraining/Results/save', exist_ok=True)
print("Python:", sys.version)
print("Torch version:", torch.__version__)
print("Torch CUDA version:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("Device count:", torch.cuda.device_count())

os.makedirs('../../save', exist_ok=True)

print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("Device count:", torch.cuda.device_count())

if __name__ == '__main__':
    start_time = time.time()

    # Define paths
    path_project = os.path.abspath('.')

    args = args_parser()

    print("[Args check]", args)
    print("[Args check] has cpu flag?", hasattr(args, "cpu"))

    import random, numpy as np, torch

    def set_seed(seed: int, deterministic: bool = False):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Full determinism
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    set_seed(args.seed, args.deterministic)

    if args.partition_type != 'iid':
        args.iid = 0
    else:
        args.iid = 1

    exp_details(args)

    # if args.gpu is not None:
    #     torch.cuda.set_device(int(args.gpu))
    # device = 'cuda' if torch.cuda.is_available() and args.gpu is not None else 'cpu'

    use_cuda = torch.cuda.is_available() and (not args.cpu)
    device = 'cuda' if use_cuda else 'cpu'

    if device == 'cuda':
        torch.cuda.set_device(args.gpu)

    # Local dataset and user groups
    train_dataset, test_dataset, user_groups, test_user_groups, label_mappings = get_dataset(args)

    logger = SummaryWriter('./logs')

    try:
        from sampling import print_label_distribution
        print_label_distribution(user_groups, train_dataset)

        # ------------------------
        # Build models (per setup)
        # ------------------------
        if args.model == 'CNN':
            if args.dataset == 'mnist':
                global_model = CNNMNIST(args=args).to(device)
            else:
                exit('Invalid dataset selected')

        elif args.model == 'AutoEncoder':
            if args.dataset == 'mnist':
                global_model = AutoEncoder().to(device)
            else:
                exit('Invalid dataset selected')

        elif args.model == 'MLP':
            if args.dataset == 'mnist':
                # Define client-specific output sizes (same as Scenario 1)
                client_output_dims = {
                    0: 2,  # Client 0: [2, not-2]
                    1: 3,  # Client 1: [1,2,none]
                    2: 2   # Client 2: [6, not-6]
                }

                # Create an independent model (encoder+head) per client
                client_models = {
                    cid: ClientModel(encoder=Encoder_model(), output_dim=out_dim).to(device)
                    for cid, out_dim in client_output_dims.items()
                }

                for cid, model in client_models.items():
                    # Last layer out_features from client's head for sanity
                    out_dim = model.head.fc[-1].out_features
                    print(f"Client {cid} - Head Output Dim: {out_dim}")
            else:
                raise ValueError('Invalid dataset selected')
        else:
            raise ValueError('Invalid model type')

        # ------------------------
        # Set models to train mode
        # ------------------------
        if args.model != 'MLP':
            global_model.to(device)
            global_model.train()
            print(global_model)
        else:
            for m in client_models.values():
                m.train()

        print(f"\nTotal users: {args.num_users}")
        for uid in user_groups:
            print(f"User {uid} has {len(user_groups[uid])} samples")

        # ------------------------
        # Tracking structures
        # ------------------------
        client_losses = {i: [] for i in range(args.num_users)}
        val_acc_list, net_list = [], []
        cv_loss, cv_acc = [], []
        print_every = 2
        val_loss_pre, counter = 0, 0

        # Per-client TRAIN histories
        client_train_acc = defaultdict(list)   # % per epoch
        client_train_loss = defaultdict(list)

        # Per-client TEST histories
        client_test_accuracies = defaultdict(list)  # % per epoch
        client_test_losses = defaultdict(list)

        # Overall histories (averaged over clients each epoch)
        overall_test_acc_history = []
        overall_test_loss_history = []

        # "Server" averages per epoch (just average over clients; no aggregation)
        server_avg_train_loss = []
        server_avg_train_acc = []

        # Also keep simple per-epoch train loss list
        train_loss = []

        test_acc, test_loss = None, None

        # Training loop (Scenario 3)
        for epoch in tqdm(range(args.epochs)):
            print(f'\n | Epoch: {epoch+1} (Scenario 3: Independent clients) |\n')

            # Pre-append NaNs so every client has exactly args.epochs elements
            for cid in range(args.num_users):
                client_train_acc[cid].append(np.nan)
                client_train_loss[cid].append(np.nan)

            # Decide which clients train this epoch
            # (you can keep "frac" behaviour like Scenario 1 or force all train)
            m = max(int(args.frac * args.num_users), 1)
            idxs_users = np.random.choice(range(args.num_users), m, replace=False)

            participated = set()

            # Client-wise training (Local training; NO aggregation)
            if args.model == 'MLP':
                for cid in idxs_users:
                    model_c = client_models[cid]
                    model_c.train()

                    print(f"Training Client {cid}")

                    local_obj = LocalUpdate(
                        args=args,
                        dataset=train_dataset,
                        idxs=user_groups[cid],
                        logger=logger,
                        encoder=model_c.encoder,
                        head=model_c.head,
                        client_id=cid,
                        label_mappings=label_mappings
                    )

                    # train this client's model in-place; ignore returned weights
                    _, loss = local_obj.update_weights(global_round=epoch)
                    client_train_loss[cid][-1] = float(loss)

                    acc_train_frac, avg_loss_train = local_obj.inference()
                    client_train_acc[cid][-1] = float(acc_train_frac) * 100.0  # store % accuracy

                    participated.add(cid)

            else:
                # CNN / AE path: keep as-is if you ever need it (not focus of Scenario 3)
                global_model.train()
                # you can add per-client logic here if you define user_groups similarly

            # per-epoch averages across clients (ignore NaNs)
            server_avg_train_loss.append(
                float(np.nanmean([client_train_loss[c][-1] for c in client_train_loss]))
            )
            server_avg_train_acc.append(
                float(np.nanmean([client_train_acc[c][-1] for c in client_train_acc]))
            )
            train_loss.append(server_avg_train_loss[-1])

            # ------------------------
            # TEST phase per epoch
            # ------------------------
            if args.model == 'MLP':
                for cid in range(args.num_users):
                    model_c = client_models[cid]
                    model_c.eval()

                    # Use the same per-client test groups as Scenario 1
                    client_test_subset = Subset(test_dataset, list(test_user_groups[cid]))

                    acc_c, loss_c = test_inference(
                        args,
                        model=model_c,
                        test_dataset=client_test_subset,
                        label_mapping=label_mappings[cid]
                    )

                    client_test_accuracies[cid].append(float(acc_c))
                    client_test_losses[cid].append(float(loss_c))

                # overall test stats (mean over clients)
                last_accs = [client_test_accuracies[c][-1] for c in client_test_accuracies]
                last_losses = [client_test_losses[c][-1] for c in client_test_losses]

                overall_test_acc_history.append(float(np.mean(last_accs)))
                overall_test_loss_history.append(float(np.mean(last_losses)))

            # Print training summary every few epochs
            if (epoch + 1) % print_every == 0:
                print(f'\nAvg Training Stats after {epoch+1} epochs:')
                print(f'Training Loss (avg over clients): {np.mean(np.array(train_loss))}')
                print('Train Accuracy (avg over clients): {:.2f}% \n'.format(server_avg_train_acc[-1]))

        # ------------------------
        # Final summary
        # ------------------------
        print(f'\n Results after {args.epochs} epochs of training (IndividualTraining):')
        print('|---- Avg Train Accuracy (last epoch mean over clients): {:.2f}%'.format(server_avg_train_acc[-1]))
        if len(overall_test_acc_history) > 0:
            print('|---- Test Accuracy (last epoch mean over clients): {:.2f}%'.format(overall_test_acc_history[-1]))
            print('|---- Test Loss (last epoch mean over clients): {:.4f}'.format(overall_test_loss_history[-1]))
        else:
            print('|---- Test Accuracy/Loss: Not computed')

        # ------------------------
        # Save train curves
        # ------------------------
        file_name = './Results/save/objects/{}_{}_{}_C[{}]_partition[{}]_E[{}]_B[{}].pkl'.\
            format(args.dataset, args.model, args.epochs, args.frac, args.partition_type, args.local_ep, args.local_bs)

        save_dir = 'Results/save'
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.dirname(file_name), exist_ok=True)

        with open(file_name, 'wb') as f:
            pickle.dump([server_avg_train_loss, server_avg_train_acc], f)

        print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

        # ------------------------
        # Plotting (IndividualTraining)
        # ------------------------
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        rounds = list(range(1, args.epochs + 1))

        # Helper
        def dict_to_df(d, rounds):
            import pandas as pd
            cols = sorted(d.keys())
            data = {f'client_{cid}': d[cid] for cid in cols}
            df = pd.DataFrame(data, index=rounds)
            df.index.name = 'round'
            return df

        # 1) Training loss (average over clients)
        plt.figure()
        plt.title('Training Loss vs Epochs - IndividualTraining')
        plt.plot(rounds, server_avg_train_loss)
        plt.xlabel('Epochs')
        plt.ylabel('Training Loss')
        plt.xticks(rounds)
        plt.savefig('./Results/save/server_avg_train_loss_per_epoch_IndividualTraining.png')
        tikzplotlib.save('Results/CommonCodeWithNoise0.5/server_avg_train_loss_per_epoch_IndividualTraining.tikz')

        # 2) Training accuracy (average over clients)
        plt.figure()
        plt.title('Average Train Accuracy vs Epochs - IndividualTraining')
        plt.plot(rounds, server_avg_train_acc)
        plt.xlabel('Epochs')
        plt.ylabel('Average Train Accuracy (%)')
        plt.xticks(rounds)
        plt.savefig('./Results/save/server_avg_train_accuracy_per_epoch_IndividualTraining.png')
        tikzplotlib.save('Results/CommonCodeWithNoise0.5/server_avg_train_accuracy_per_epoch_IndividualTraining.tikz')

        # 3) Overall TEST accuracy (mean over clients)
        if len(overall_test_acc_history) > 0:
            plt.figure()
            plt.title('Overall TEST Accuracy vs Epochs - IndividualTraining')
            plt.plot(rounds, overall_test_acc_history)
            plt.xlabel('Epochs')
            plt.ylabel('Test Accuracy (%)')
            plt.xticks(rounds)
            plt.savefig('./Results/save/overall_test_accuracy_per_epoch_IndividualTraining.png')
            tikzplotlib.save(f'{save_dir}/overall_test_accuracy_per_epoch_IndividualTraining.tikz')

        # 4) Per-client TRAIN accuracy
        import pandas as pd
        df_train_acc = dict_to_df(client_train_acc, rounds)
        df_train_acc.to_csv(f'{save_dir}/client_train_accuracy_per_epoch_IndividualTraining.csv')

        plt.figure()
        plt.title('Client-wise TRAIN Accuracy per Epoch - IndividualTraining')
        for cid in sorted(client_train_acc.keys()):
            plt.plot(rounds, client_train_acc[cid], label=f'Client {cid}')
        plt.xlabel('Epochs')
        plt.ylabel('Train Accuracy (%)')
        plt.xticks(rounds)
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(f'{save_dir}/client_train_accuracy_per_epoch_IndividualTraining.png')
        tikzplotlib.save(f'{save_dir}/client_train_accuracy_per_epoch_IndividualTraining.tikz')

        # 5) Per-client TRAIN loss
        df_train_loss = dict_to_df(client_train_loss, rounds)
        df_train_loss.to_csv(f'{save_dir}/client_train_loss_per_epoch_IndividualTraining.csv')

        plt.figure()
        plt.title('Client-wise TRAIN Loss per Epoch - IndividualTraining')
        for cid in sorted(client_train_loss.keys()):
            plt.plot(rounds, client_train_loss[cid], label=f'Client {cid}')
        plt.xlabel('Epochs')
        plt.ylabel('Train Loss')
        plt.xticks(rounds)
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(f'{save_dir}/client_train_loss_per_epoch_IndividualTraining.png')
        tikzplotlib.save(f'{save_dir}/client_train_loss_per_epoch_IndividualTraining.tikz')

        # 6) Per-client TEST accuracy
        if len(client_test_accuracies) > 0:
            df_test_acc = dict_to_df(client_test_accuracies, rounds)
            df_test_acc.to_csv(f'{save_dir}/client_test_accuracy_per_epoch_IndividualTraining.csv')

            plt.figure()
            plt.title('Client-wise TEST Accuracy per Epoch - IndividualTraining')
            for cid in sorted(client_test_accuracies.keys()):
                plt.plot(rounds, client_test_accuracies[cid], label=f'Client {cid}')
            plt.xlabel('Epochs')
            plt.ylabel('Test Accuracy (%)')
            plt.xticks(rounds)
            plt.legend(loc='lower right')
            plt.tight_layout()
            plt.savefig(f'{save_dir}/client_test_accuracy_per_epoch_IndividualTraining.png')
            tikzplotlib.save(f'{save_dir}/client_test_accuracy_per_epoch_IndividualTraining.tikz')

        # 7) Per-client TEST loss
        if len(client_test_losses) > 0:
            df_test_loss = dict_to_df(client_test_losses, rounds)
            df_test_loss.to_csv(f'{save_dir}/client_test_loss_per_epoch_IndividualTraining.csv')

            plt.figure()
            plt.title('Client-wise TEST Loss per Epoch - IndividualTraining')
            for cid in sorted(client_test_losses.keys()):
                plt.plot(rounds, client_test_losses[cid], label=f'Client {cid}')
            plt.xlabel('Epochs')
            plt.ylabel('Test Loss')
            plt.xticks(rounds)
            plt.legend(loc='lower right')
            plt.tight_layout()
            plt.savefig(f'{save_dir}/client_test_loss_per_epoch_IndividualTraining.png')
            tikzplotlib.save(f'{save_dir}/client_test_loss_per_epoch_IndividualTraining.tikz')

        # 8) Overall TEST loss (mean over clients)
        if len(overall_test_loss_history) > 0:
            plt.figure()
            plt.title('Overall TEST Loss vs Epochs - IndividualTraining')
            plt.plot(rounds, overall_test_loss_history)
            plt.xlabel('Epochs')
            plt.ylabel('Test Loss')
            plt.xticks(rounds)
            plt.savefig(f'{save_dir}/overall_test_loss_per_epoch_IndividualTraining.png')
            tikzplotlib.save(f'{save_dir}/overall_test_loss_per_epoch_IndividualTraining.tikz')

    finally:
        logger.close()


