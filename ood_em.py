"""Run Experiment

This script allows to run one federated learning experiment; the experiment name, the method and the
number of clients/tasks should be precised along side with the hyper-parameters of the experiment.

The results of the experiment (i.e., training logs) are written to ./logs/ folder.

This file can also be imported as a module and contains the following function:

    * run_experiment - runs one experiments given its arguments
"""
import os.path
import torch.nn as nn
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

from utils.args import *
from utils.utils import *


class FEMNIST(object):
    def __init__(self):
        kwargs = {'num_workers': 1, 'pin_memory': True}
        train_dataset = datasets.MNIST('data/mnist', train=True, download=True, transform=transforms.ToTensor())
        indices = (train_dataset.targets != 1).nonzero().reshape([-1])
        train_dataset = torch.utils.data.Subset(train_dataset, indices)
        self.train_loader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_size=64, shuffle=True, **kwargs)

        test_dataset = datasets.MNIST('data/mnist', train=False, download=True, transform=transforms.ToTensor())
        indices = (test_dataset.targets != 1).nonzero().reshape([-1])
        test_dataset = torch.utils.data.Subset(test_dataset, indices)
        self.test_loader = torch.utils.data.DataLoader(test_dataset,
                                                       batch_size=64, shuffle=True, **kwargs)

        ood_dataset = datasets.MNIST('data/mnist', train=False, download=True, transform=transforms.ToTensor())
        indices = (ood_dataset.targets == 1).nonzero().reshape([-1])
        ood_dataset = torch.utils.data.Subset(ood_dataset, indices)
        self.ood_loader = torch.utils.data.DataLoader(ood_dataset,
                                                      batch_size=64, shuffle=True, **kwargs)



def init_clients(args_, root_path, logs_dir, save_path):
    """
    initialize clients from data folders
    :param args_:
    :param root_path: path to directory containing data folders
    :param logs_dir: path to logs root
    :return: List[Client]
    """
    print("===> Building data iterators..")
    train_iterators, val_iterators, test_iterators = \
        get_loaders(
            type_=LOADER_TYPE[args_.experiment],
            root_path=root_path,
            batch_size=args_.bz,
            is_validation=args_.validation
        )

    print("===> Initializing clients..")
    clients_ = []
    for task_id, (train_iterator, val_iterator, test_iterator) in \
            enumerate(tqdm(zip(train_iterators, val_iterators, test_iterators), total=len(train_iterators))):

        if train_iterator is None or test_iterator is None:
            continue

        learners_ensemble = \
            get_learners_ensemble(
                n_learners=args_.n_learners,
                client_type=CLIENT_TYPE[args_.method],
                name=args_.experiment,
                device=args_.device,
                optimizer_name=args_.optimizer,
                scheduler_name=args_.lr_scheduler,
                initial_lr=args_.lr,
                input_dim=args_.input_dimension,
                output_dim=args_.output_dimension,
                n_rounds=args_.n_rounds,
                seed=args_.seed,
                mu=args_.mu,
                embedding_dim=args_.embedding_dimension
            )

        logs_path = os.path.join(logs_dir, "task_{}".format(task_id))
        os.makedirs(logs_path, exist_ok=True)
        logger = SummaryWriter(logs_path)

        save_path = os.path.join(save_path, "task_{}".format(task_id))

        client = get_client(
            client_type=CLIENT_TYPE[args_.method],
            learners_ensemble=learners_ensemble,
            q=args_.q,
            train_iterator=train_iterator,
            val_iterator=val_iterator,
            test_iterator=test_iterator,
            logger=logger,
            local_steps=args_.local_steps,
            save_path=save_path,
            tune_locally=args_.locally_tune_clients
        )

        clients_.append(client)

    return clients_


def rand_label(n_labels, targets):
    for i in range(targets.size(0)):
        y = targets[i]
        a = np.random.randint(low=1, high=9)
        y_ = (y + a) % 10
        targets[i] = y_

    return targets


def run_experiment(args_):
    torch.manual_seed(args_.seed)

    data_dir = get_data_dir(args_.experiment)
    data_dir1 = get_data_dir(args_.experiment, ood=True)
    save_dir = get_save_dir(args_.experiment)

    if "logs_dir" in args_:
        logs_dir = args_.logs_dir
    else:
        logs_dir = os.path.join("logs", args_to_string(args_))

    train_iterators, val_iterators, test_iterators = \
        get_loaders(
            type_=LOADER_TYPE[args_.experiment],
            root_path=os.path.join(data_dir, "train"),
            batch_size=args_.bz,
            is_validation=args_.validation,
            dist_shift=args.dist_shift,
            dp=args.dp,
            emb_size=args_.embedding_dimension
        )

    train_iterators_ood, val_iterators_ood, test_iterators_ood = \
        get_loaders(
            type_=LOADER_TYPE[args_.experiment],
            root_path=os.path.join(data_dir1, "train"),
            batch_size=args_.bz,
            is_validation=args_.validation,
            dist_shift=args.dist_shift,
            dp=args.dp,
            emb_size=args_.embedding_dimension
        )

    ensemble = \
        get_learners_ensemble(
            n_learners=args_.n_learners,
            client_type=CLIENT_TYPE[args_.method],
            name=args_.experiment,
            n_gmm=3,
            device=args_.device,
            optimizer_name=args_.optimizer,
            scheduler_name=args_.lr_scheduler,
            initial_lr=args_.lr,
            input_dim=args_.input_dimension,
            output_dim=args_.output_dimension,
            n_rounds=args_.n_rounds,
            seed=args_.seed,
            mu=args_.mu,
            embedding_dim=args_.embedding_dimension
        )

    dir_path = '/home/zshuai8/fed_learning/FedGMM/saves/femnist_ood/FedEM_dist_shift_lr_003_gc_3_emb_48_ood/'
    for learner_id, learner in enumerate(ensemble):
        chkpts_path = os.path.join(dir_path, f"chkpts_{learner_id}.pt")
        learner.model.load_state_dict(torch.load(chkpts_path))

    import numpy as np
    weights = np.load('/home/zshuai8/fed_learning/FedGMM/saves/femnist_ood/FedEM_dist_shift_lr_003_gc_3_emb_48_ood/train_client_weights.npy')
    avg_weights = np.mean(weights,axis=0)
    ensemble.learners_weights = avg_weights
    dic = dict()

    with torch.no_grad():
        y_x_nll = []
        for i, data_loader in enumerate(test_iterators):
            for data, label,_ in data_loader:
                data = data.float().to(ensemble.device)
                # recon_batch = ensemble.autoencoder.model.encode(data)

                predictive_log_likelihood = evalute_batch(ensemble.learners,ensemble.learners_weights, (data, label), ensemble.device)
                # x_nll.append(predictive_log_likelihood)
                y_x_nll.append(predictive_log_likelihood)

        # x_nll = torch.cat(x_nll).cpu().numpy()
        y_x_nll = torch.cat(y_x_nll).cpu().numpy()
        # dic['test:x'] = x_nll
        dic['test:y'] = y_x_nll

        y_x_nll2 = []
        for i, data_loader in enumerate(test_iterators_ood):
            for data, label, _ in data_loader:
                data = data.float().to(ensemble.device)

                predictive_log_likelihood = evalute_batch(ensemble.learners,ensemble.learners_weights, (data, label), ensemble.device)
                y_x_nll2.append(predictive_log_likelihood)


        y_x_nll2 = torch.cat(y_x_nll2).cpu().numpy()
        dic['ood:y'] = y_x_nll2
        from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_recall_curve
        overall_y = np.concatenate((y_x_nll, y_x_nll2))
        overall_y = (overall_y - np.min(overall_y)) / (np.max(overall_y) - np.min(overall_y))

        labels = np.concatenate((np.ones(len(y_x_nll)), np.zeros(len(y_x_nll2))))


        roc_auc = roc_auc_score(labels, overall_y)
        print(roc_auc)

        # Compute AP
        ap = average_precision_score(labels, overall_y, average='macro')
        print("AP:", ap)

        precision, recall, thresholds = precision_recall_curve(labels, overall_y)
        f1_scores = 2 * recall * precision / (recall + precision)
        print('Best threshold: ', thresholds[np.argmax(f1_scores)])
        print('Best F1-Score: ', np.max(f1_scores))

    torch.save(dic, 'FedEM_ood.pt')

    # if args_.decentralized:
    #     aggregator_type = 'decentralized'
    # else:
    #     aggregator_type = AGGREGATOR_TYPE[args_.method]

    # aggregator =\
    #     get_aggregator(
    #         aggregator_type=aggregator_type,
    #         clients=clients,
    #         global_learners_ensemble=global_learners_ensemble,
    #         lr_lambda=args_.lr_lambda,
    #         lr=args_.lr,
    #         q=args_.q,
    #         mu=args_.mu,
    #         communication_probability=args_.communication_probability,
    #         sampling_rate=args_.sampling_rate,
    #         log_freq=args_.log_freq,
    #         global_train_logger=global_train_logger,
    #         global_test_logger=global_test_logger,
    #         test_clients=test_clients,
    #         verbose=args_.verbose,
    #         seed=args_.seed
    #     )

    # if "save_dir" in args_:
    #     save_dir = os.path.join(a rgs_.save_dir)
    #
    #     os.makedirs(save_dir, exist_ok=True)
    #     aggregator.save_state(save_dir)
    #
    # print("Training..")
    # pbar = tqdm(total=args_.n_rounds)
    # current_round = 0
    # while current_round <= args_.n_rounds:
    #
    #     aggregator.mix()
    #
    #     if current_round % 10 == 0:
    #         aggregator.save_state(save_dir)
    #
    #     if aggregator.c_round != current_round:
    #         pbar.update(1)
    #         current_round = aggregator.c_round

def evalute_batch(learners, learners_weights, batch, device):
    criterion = nn.NLLLoss(reduction="none")
    for learner in learners:
        learner.model.eval()
    import torch
    n_samples = 0
    with torch.no_grad():
        x, y = batch
        x = x.to(device).type(torch.float32)
        y = y.to(device)

        with torch.no_grad():
            # for x, y in batch:
            x = x.to(device).type(torch.float32)
            y = y.to(device)
            n_samples += y.size(0)

            y_pred = 0.
            for learner_id, learner in enumerate(learners):

                y_pred += learners_weights[learner_id] * F.softmax(learner.model(x), dim=1)

            y_pred = torch.clamp(y_pred, min=0., max=1.)

        y_pred = torch.clamp(y_pred, min=0., max=1.)


        assert not torch.isnan(criterion(torch.log(y_pred), y).sum())
        losses = criterion(torch.log(y_pred), y)



    return -losses

if __name__ == "__main__":
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    args = parse_args()
    run_experiment(args)
