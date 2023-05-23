"""Run Experiment

This script allows to run one federated learning experiment; the experiment name, the method and the
number of clients/tasks should be precised along side with the hyper-parameters of the experiment.

The results of the experiment (i.e., training logs) are written to ./logs/ folder.

This file can also be imported as a module and contains the following function:

    * run_experiment - runs one experiments given its arguments
"""
import os.path

from torch.utils.tensorboard import SummaryWriter

from utils.args import *
from utils.utils import *
import pickle

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
            is_validation=args_.validation,
            dist_shift=args.dist_shift,
            dp = args.dp,
            emb_size= args_.embedding_dimension
        )


    # all_data_tensor = []
    # for cur_data in train_iterators:
    #     all_data_tensor.append(cur_data.dataset.data)
    # all_data_tensor = torch.cat(all_data_tensor, dim=0)
    #
    # model = models.resnet18(pretrained=True)
    #
    # del model.fc
    # all_data_tensor = all_data_tensor.view(-1,1,28,28)
    # x = all_data_tensor
    # if all_data_tensor.shape[1] == 1:
    #     x = all_data_tensor.repeat(1, 3, 1, 1)
    # x = model.conv1(x.float())
    # x = model.bn1(x)
    # x = model.relu(x)
    # x = model.maxpool(x)
    #
    # x = model.layer1(x)
    # x = model.layer2(x)
    # x = model.layer3(x)
    # x = model.layer4(x)
    #
    # # Extract the feature maps produced by the encoder
    # encoder_output = x.squeeze()
    # U, S, V = torch.svd(encoder_output)
    # global PCA_V
    # PCA_V = V
    # print(PCA_V.size())
    # with open(f"data/mnist9/all_data/PCA.pkl" , 'wb') as f:
    #     pickle.dump(PCA_V, f)
    # raise


    # encoder_output = encoder_output.view(encoder_output.size(0), -1)
    # pca_transformer = PCA(n_components=emb_size)
    # # Fit the PCA transformer to your data
    #
    # X_pca = pca_transformer.fit_transform(encoder_output.detach().numpy())
    # # Convert the resulting principal components to a PyTorch tensor
    # projected = torch.from_numpy(X_pca).float().cuda()

    print("===> Initializing clients..")
    clients_ = []
    for task_id, (train_iterator, val_iterator, test_iterator) in \
            enumerate(tqdm(zip(train_iterators, val_iterators, test_iterators), total=len(train_iterators))):
        # if train_iterator is None or test_iterator is None:
        #     continue
        # if
        learners_ensemble =\
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
                embedding_dim=args_.embedding_dimension,
                n_gmm=args_.n_gmm
            )
        # learners_ensemble.load_state("saves/femnist_unseen_base/FedGMM_sanity/global_ensemble.pt")
        logs_path = os.path.join(logs_dir, "task_{}".format(task_id))
        os.makedirs(logs_path, exist_ok=True)
        logger = SummaryWriter(logs_path)

        save_path_ = os.path.join(save_path, "task_{}".format(task_id))
        client = get_client(
            client_type=CLIENT_TYPE[args_.method],
            learners_ensemble=learners_ensemble,
            q=args_.q,
            train_iterator=train_iterator,
            val_iterator=val_iterator,
            test_iterator=test_iterator,
            logger=logger,
            local_steps=args_.local_steps,
            save_path=save_path_,
            tune_locally=args_.locally_tune_clients
        )

        clients_.append(client)
    return clients_


def run_experiment(args_):
    torch.manual_seed(args_.seed)

    data_dir = get_data_dir(args_.experiment, unseen=False)
    save_dir = get_save_dir(args_.experiment)

    if "logs_dir" in args_:
        logs_dir = args_.logs_dir
    else:
        logs_dir = os.path.join("logs", args_to_string(args_))

    print("==> Clients initialization..")
    clients = init_clients(args_,
                           root_path=os.path.join(data_dir, "train"),
                           logs_dir=os.path.join(logs_dir, "train"),
                           save_path=os.path.join(save_dir, "train"),

                           )

    print("==> Test Clients initialization..")
    test_clients = init_clients(args_, root_path=os.path.join(data_dir, "test"),
                                logs_dir=os.path.join(logs_dir, "test"),
                                save_path=os.path.join(save_dir, "test"))

    logs_path = os.path.join(logs_dir, "train", "global")
    os.makedirs(logs_path, exist_ok=True)
    global_train_logger = SummaryWriter(logs_path)
    logs_path = os.path.join(logs_dir, "test", "global")
    os.makedirs(logs_path, exist_ok=True)
    global_test_logger = SummaryWriter(logs_path)
    global_learners_ensemble = \
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
            embedding_dim=args_.embedding_dimension,
            n_gmm=args_.n_gmm,
        )

    if args_.decentralized:
        aggregator_type = 'decentralized'
    else:
        aggregator_type = AGGREGATOR_TYPE[args_.method]

    # global_learners_ensemble.load_state("saves/femnist_unseen_base/FedGMM_sanity/global_ensemble.pt")
    aggregator =\
        get_aggregator(
            aggregator_type=aggregator_type,
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            lr_lambda=args_.lr_lambda,
            lr=args_.lr,
            q=args_.q,
            mu=args_.mu,
            communication_probability=args_.communication_probability,
            sampling_rate=args_.sampling_rate,
            log_freq=args_.log_freq,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            test_clients=test_clients,
            verbose=args_.verbose,
            em_step=args.em_step,
            seed=args_.seed
        )
    if "save_dir" in args_:
        save_dir = os.path.join(args_.save_dir)

        os.makedirs(save_dir, exist_ok=True)
        aggregator.save_state(save_dir)

    print("Training..")
    pbar = tqdm(total=args_.n_rounds)
    current_round = 0
    gmm = False
    while current_round <= args_.n_rounds:
        # aggregator.update_clients()
        # aggregator.write_log()
        # raise
        if aggregator_type == "ACGcentralized":
            aggregator.mix()
        else:
            aggregator.mix()
        if current_round % 10 == 0:
            aggregator.save_state(save_dir)
            print("saved at epoch", current_round)

        if aggregator.c_round != current_round:
            pbar.update(1)
            current_round = aggregator.c_round
        print(current_round)
        # if current_round > 50:
        #     gmm=False

    global_train_logger.flush()
    global_test_logger.flush()
    global_train_logger.close()
    global_test_logger.close()



if __name__ == "__main__":
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    args = parse_args()
    run_experiment(args)
