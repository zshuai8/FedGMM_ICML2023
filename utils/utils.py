import torch.nn
from tqdm import tqdm
from sklearn.decomposition import PCA
from aggregator import *
from client import *
from datasets import *
from learners.autoencoder import *
from resnet import *

from learners.learner import *
from learners.learners_ensemble import *
from models import *
from .constants import *
from .decentralized import *
from .metrics import *
from .optim import *
from torch.utils.data import DataLoader
from torchvision import models



def get_data_dir(experiment_name, ood=False, unseen=False):
    """
    returns a string representing the path where to find the datafile corresponding to the experiment

    :param experiment_name: name of the experiment
    :return: str

    """
    if ood:
        return os.path.join("data", experiment_name, "all_data_ood_t")
    if unseen:
        return os.path.join("data", experiment_name, "all_data_unseen")
    data_dir = os.path.join("data", experiment_name, "all_data")

    return data_dir


def get_save_dir(experiment_name):
    """
    returns a string representing the path where to find the datafile corresponding to the experiment

    :param experiment_name: name of the experiment
    :return: str

    """
    data_dir = os.path.join("saves", experiment_name)

    return data_dir

def get_learner(
        name,
        device,
        optimizer_name,
        scheduler_name,
        initial_lr,
        mu,
        n_rounds,
        seed,
        embed_dim,
        input_dim=None,
        output_dim=None
):
    """
    constructs the learner corresponding to an experiment for a given seed

    :param name: name of the experiment to be used; possible are
                 {`synthetic`, `cifar10`, `emnist`, `shakespeare`}
    :param device: used device; possible `cpu` and `cuda`
    :param optimizer_name: passed as argument to utils.optim.get_optimizer
    :param scheduler_name: passed as argument to utils.optim.get_lr_scheduler
    :param initial_lr: initial value of the learning rate
    :param mu: proximal term weight, only used when `optimizer_name=="prox_sgd"`
    :param input_dim: input dimension, only used for synthetic dataset
    :param output_dim: output_dimension; only used for synthetic dataset
    :param n_rounds: number of training rounds, only used if `scheduler_name == multi_step`, default is None;
    :param seed:
    :return: Learner

    """
    torch.manual_seed(seed)

    if name == "synthetic" or name == "gmsynthetic":
        if output_dim == 2:
            criterion = nn.BCEWithLogitsLoss(reduction="none").to(device)
            metric = binary_accuracy
            model = LinearLayer(input_dim, 1).to(device)
            is_binary_classification = True
        else:
            criterion = nn.CrossEntropyLoss(reduction="none").to(device)
            metric = accuracy
            model = LinearLayer(input_dim, output_dim).to(device)
            is_binary_classification = False
    elif name == "cifar10":
        criterion = nn.CrossEntropyLoss(reduction="none").to(device)
        metric = accuracy
        # model =  nn.Linear(embed_dim, 10).to(device)
        model = get_mobilenet(n_classes=10).to(device)
        # model = get_resnet18(n_classes=10).to(device)
        is_binary_classification = False
    elif name == "cifar100":
        criterion = nn.CrossEntropyLoss(reduction="none").to(device)
        metric = accuracy
        # model = nn.Linear(embed_dim, 100).to(device)
        model = get_mobilenet(n_classes=100).to(device)
        is_binary_classification = False
    elif name == "emnist" or name == "femnist" or name == "emnist_r":
        criterion = nn.CrossEntropyLoss(reduction="none").to(device)
        metric = accuracy
        model = FemnistCNN(num_classes=62, pretrain=False).to(device)
        is_binary_classification = False
    elif name == "mnist":
        criterion = nn.CrossEntropyLoss(reduction="none").to(device)
        metric = accuracy
        model = FemnistCNN(num_classes=10, pretrain=False).to(device)
        is_binary_classification = False
    elif name == "mnist9":
        criterion = nn.CrossEntropyLoss(reduction="none").to(device)
        metric = accuracy
        model = FemnistCNN(num_classes=10, pretrain=False).to(device)
        is_binary_classification = False
    elif name == "mnist_rep":
        criterion = nn.CrossEntropyLoss(reduction="none").to(device)
        metric = accuracy
        model = FC_Classifier(embedding_size=embed_dim, num_class=10).to(device)
        is_binary_classification = False
    elif name == "shakespeare":
        all_characters = string.printable
        labels_weight = torch.ones(len(all_characters), device=device)
        for character in CHARACTERS_WEIGHTS:
            labels_weight[all_characters.index(character)] = CHARACTERS_WEIGHTS[character]
        labels_weight = labels_weight * 8

        criterion = nn.CrossEntropyLoss(reduction="none", weight=labels_weight).to(device)
        metric = accuracy
        model = \
            NextCharacterLSTM(
                input_size=SHAKESPEARE_CONFIG["input_size"],
                embed_size=SHAKESPEARE_CONFIG["embed_size"],
                hidden_size=SHAKESPEARE_CONFIG["hidden_size"],
                output_size=SHAKESPEARE_CONFIG["output_size"],
                n_layers=SHAKESPEARE_CONFIG["n_layers"]
            ).to(device)
        is_binary_classification = False

    else:
        raise NotImplementedError

    optimizer = \
        get_optimizer(
            optimizer_name=optimizer_name,
            model=model,
            lr_initial=initial_lr,
            mu=mu
        )
    lr_scheduler = \
        get_lr_scheduler(
            optimizer=optimizer,
            scheduler_name=scheduler_name,
            n_rounds=n_rounds
        )

    if name == "shakespeare":
        return LanguageModelingLearner(
            model=model,
            criterion=criterion,
            metric=metric,
            device=device,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            is_binary_classification=is_binary_classification
        )
    else:
        return Learner(
            model=model,
            criterion=criterion,
            metric=metric,
            device=device,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            is_binary_classification=is_binary_classification
        )


global_ac = None
def get_learners_ensemble(
        n_learners,
        client_type,
        name,
        device,
        optimizer_name,
        scheduler_name,
        initial_lr,
        mu,
        n_rounds,
        seed,
        n_gmm,
        input_dim=None,
        output_dim=None,
        embedding_dim=None):
    """
    constructs the learner corresponding to an experiment for a given seed

    :param n_learners: number of learners in the ensemble
    :param name: name of the experiment to be used; possible are
                 {`synthetic`, `cifar10`, `emnist`, `shakespeare`}
    :param device: used device; possible `cpu` and `cuda`
    :param optimizer_name: passed as argument to utils.optim.get_optimizer
    :param scheduler_name: passed as argument to utils.optim.get_lr_scheduler
    :param initial_lr: initial value of the learning rate
    :param mu: proximal term weight, only used when `optimizer_name=="prox_sgd"`
    :param input_dim: input dimension, only used for synthetic dataset
    :param output_dim: output_dimension; only used for synthetic dataset
    :param n_rounds: number of training rounds, only used if `scheduler_name == multi_step`, default is None;
    :param seed:
    :return: LearnersEnsemble

    """

    global global_ac
    learners = [
        get_learner(
            name=name,
            device=device,
            optimizer_name=optimizer_name,
            scheduler_name=scheduler_name,
            initial_lr=initial_lr,
            input_dim=input_dim,
            output_dim=output_dim,
            n_rounds=n_rounds,
            seed=seed + learner_id,
            mu=mu,
            embed_dim=embedding_dim
        ) for learner_id in range(n_learners)
    ]

    learners_weights = torch.ones(n_learners) / n_learners
    if client_type == "ACGmixture":
        if name == "mnist" or name == "emnist" or name == "femnist" or name == "mnist9" or name == "emnist_r":
            assert embedding_dim is not None, "Embedding dimension not specified!!"
            model = resnet_pca(embedding_size=embedding_dim, name=name,input_size=(1, 28, 28))
            ckpt = 'AE_emnist.pt'
            if name == "mnist9":
                ckpt = 'AE_MNIST1.pt'
            if global_ac == None:
                global_ac = Autoencoder(
                    model=model,
                    checkpoint=None,
                    criterion=torch.nn.BCELoss(reduction='none'),
                    device=learners[0].device,
                    optimizer=get_optimizer(optimizer_name='adam', model=model, lr_initial=1e-5),
                    lr_scheduler=None
                )
                # global_ac  = models.resnet50(pretrained=True)
                global_ac.freeze()
            # ac = Autoencoder(
            #     model=model,
            #     checkpoint='AE_emnist.pt',
            #     device=learners[0].device,
            #     optimizer=get_optimizer(optimizer_name='adam', model=model, lr_initial=1e-5),
            #     lr_scheduler=None
            # )
            ac = global_ac
            return ACGLearnersEnsemble(
                learners=learners,
                embedding_dim=embedding_dim,
                autoencoder=ac,
                n_gmm=n_gmm
            )
        elif name == 'synthetic' or name == 'gmsynthetic':
            assert embedding_dim is not None, "Embedding dimension not specified!!"
            model = IDnetwork(embedding_size=input_dim)
            if global_ac == None:
                global_ac = Autoencoder(
                    model=model,
                    checkpoint=None,
                    criterion=torch.nn.MSELoss(reduction='none'),
                    device=learners[0].device,
                    optimizer=get_optimizer(optimizer_name='adam', model=model, lr_initial=1e-5),
                    lr_scheduler=None
                )
                # global_ac = models.resnet50(pretrained=True)
                global_ac.freeze()
            # ac = Autoencoder(
            #     model=model,
            #     checkpoint='AE_emnist.pt',
            #     device=learners[0].device,
            #     optimizer=get_optimizer(optimizer_name='adam', model=model, lr_initial=1e-5),
            #     lr_scheduler=None
            # )
            ac = global_ac
            return ACGLearnersEnsemble(
                learners=learners,
                embedding_dim=embedding_dim,
                autoencoder=ac,
                n_gmm=n_gmm
            )
        elif name == "cifar10" or name == 'cifar100':
            # Resnet_PCA
            # model = Resnet_PCA(embedding_size=embedding_dim, input_size=(3, 32, 32))
            # model = resnet_pca(embedding_size=embedding_dim, name=name, input_size=(3, 32, 32))
            # model = cACnetwork(embedding_size=embedding_dim, input_size=(3, 32, 32))
            model = resnet_pca(embedding_size=embedding_dim, name=name, input_size=(3, 32, 32))
            if global_ac == None:
                global_ac = Autoencoder(
                    model=model,
                    # checkpoint='AE_CIFAR10.pt',
                    checkpoint=None,
                    criterion=torch.nn.BCELoss(reduction='none'),
                    device=learners[0].device,
                    optimizer=get_optimizer(optimizer_name='adam', model=model, lr_initial=1e-5),
                    lr_scheduler=None
                )
                # global_ac = Autoencoder(
                #     model=model,
                #     checkpoint=None,
                #     criterion=torch.nn.BCELoss(reduction='none'),
                #     device=learners[0].device,
                #     optimizer=get_optimizer(optimizer_name='adam', model=model, lr_initial=1e-5),
                #     lr_scheduler=None
                # )
                # global_ac = models.resnet50(pretrained=True)
                global_ac.freeze()
            # ac = Autoencoder(
            #     model=model,
            #     checkpoint='AE_emnist.pt',
            #     device=learners[0].device,
            #     optimizer=get_optimizer(optimizer_name='adam', model=model, lr_initial=1e-5),
            #     lr_scheduler=None
            # )
            ac = global_ac
            return ACGLearnersEnsemble(
                learners=learners,
                embedding_dim=embedding_dim,
                autoencoder=ac,
                n_gmm=n_gmm
            )
        else:
            raise NotImplementedError('Experiment setting not implemented yet.')
    else:
        if name == "shakespeare":
            return LanguageModelingLearnersEnsemble(learners=learners, learners_weights=learners_weights)
        else:
            return LearnersEnsemble(learners=learners, learners_weights=learners_weights)

class FEMNIST_T_Dataset(Dataset):
  def __init__(self, samples, transform=None):
    self.samples = samples
    self.transform = transform

  def __len__(self):
    return len(self.samples)

  def __getitem__(self, idx):
    sample = self.samples[idx]
    if self.transform:
      sample = self.transform(sample)
    return sample
def get_loaders(type_, root_path, batch_size, is_validation, dist_shift, dp=False, emb_size=64):
    """
    constructs lists of `torch.utils.DataLoader` object from the given files in `root_path`;
     corresponding to `train_iterator`, `val_iterator` and `test_iterator`;
     `val_iterator` iterates on the same dataset as `train_iterator`, the difference is only in drop_last

    :param type_: type of the dataset;
    :param root_path: path to the data folder
    :param batch_size:
    :param is_validation: (bool) if `True` validation part is used as test
    :return:
        train_iterator, val_iterator, test_iterator
        (List[torch.utils.DataLoader], List[torch.utils.DataLoader], List[torch.utils.DataLoader])

    """
    if type_ == "cifar10":
        inputs, targets = get_cifar10(dist_shift, dp)
    elif type_ == "cifar100":
        inputs, targets = get_cifar100(dist_shift, dp)

    elif type_ == "emnist":
        inputs, targets = get_emnist(dist_shift, dp)
    elif type_ == "emnist_r":
        inputs, targets = get_emnist(dist_shift, dp)
    elif type_ == "mnist":
        inputs, targets = get_mnist(dist_shift, dp)
    elif type_ == "mnist9":
        inputs, targets = get_mnist9(dist_shift, dp)
    elif type_ == "mnist_rep":
        inputs, targets = get_mnist_rep()
    else:
        inputs, targets = None, None

    train_iterators, val_iterators, test_iterators = [], [], []

    for task_id, task_dir in enumerate(tqdm(os.listdir(root_path))):
        task_data_path = os.path.join(root_path, task_dir)
        train_iterator = \
            get_loader(
                type_=type_,
                path=os.path.join(task_data_path, f"train{EXTENSIONS[type_]}"),
                batch_size=batch_size,
                inputs=inputs,
                targets=targets,
                train=True,
                emb_size=emb_size
            )

        val_iterator = \
            get_loader(
                type_=type_,
                path=os.path.join(task_data_path, f"train{EXTENSIONS[type_]}"),
                batch_size=batch_size,
                inputs=inputs,
                targets=targets,
                train=False,
                emb_size=emb_size
            )

        if is_validation:
            test_set = "val"
        else:
            test_set = "test"

        test_iterator = \
            get_loader(
                type_=type_,
                path=os.path.join(task_data_path, f"{test_set}{EXTENSIONS[type_]}"),
                batch_size=batch_size,
                inputs=inputs,
                targets=targets,
                train=False
            )


        train_iterators.append(train_iterator)
        val_iterators.append(val_iterator)
        test_iterators.append(test_iterator)

    return train_iterators, val_iterators, test_iterators


def get_loader(type_, path, batch_size, train,dist_shift=True, dp=True, inputs=None, targets=None, emb_size=64):
    """
    constructs a torch.utils.DataLoader object from the given path

    :param type_: type of the dataset; possible are `tabular`, `images` and `text`
    :param path: path to the data file
    :param batch_size:
    :param train: flag indicating if train loader or test loader
    :param inputs: tensor storing the input data; only used with `cifar10`, `cifar100` and `emnist`; default is None
    :param targets: tensor storing the labels; only used with `cifar10`, `cifar100` and `emnist`; default is None
    :return: torch.utils.DataLoader

    """
    if type_ == "tabular":
        dataset = TabularDataset(path)
    elif type_ == "cifar10":
        dataset = SubCIFAR10(path, cifar10_data=inputs, cifar10_targets=targets, emb_size=emb_size)
    elif type_ == "cifar100":
        dataset = SubCIFAR100(path, cifar100_data=inputs, cifar100_targets=targets, emb_size=emb_size)
    elif type_ == "emnist":
        dataset = SubEMNIST(path, emnist_data=inputs, emnist_targets=targets)
    elif type_ == "emnist_r":
        dataset = SubEMNIST(path, emnist_data=inputs, emnist_targets=targets)
    elif type_ == "mnist":
        dataset = SubMNIST(path, emnist_data=inputs, emnist_targets=targets)
    elif type_ == "mnist9":
        dataset = SubMNIST9(path, emnist_data=inputs, emnist_targets=targets)
    elif type_ == "femnist":
        dataset = SubFEMNIST(path)
    elif type_ == "mnist_rep":
        dataset = SubMNIST_REP(path)
    elif type_ == "shakespeare":
        dataset = CharacterDataset(path, chunk_len=SHAKESPEARE_CONFIG["chunk_len"])
    else:
        raise NotImplementedError(f"{type_} not recognized type; possible are {list(LOADER_TYPE.keys())}")

    if len(dataset) == 0:
        return

    # drop last batch, because of BatchNorm layer used in mobilenet_v2
    # drop_last = ((type_ == "cifar100") or (type_ == "cifar10")) and (len(dataset) > batch_size) and train
    drop_last = (len(dataset) > batch_size) and train

    return DataLoader(dataset, batch_size=batch_size, shuffle=train, drop_last=drop_last)


def get_client(
        client_type,
        learners_ensemble,
        q,
        train_iterator,
        val_iterator,
        test_iterator,
        logger,
        local_steps,
        tune_locally,
        save_path,
        gmm_iterator_train=None,
        gmm_iterator_val=None,
        gmm_iterator_test=None
):
    """

    :param client_type:
    :param learners_ensemble:
    :param q: fairness hyper-parameter, ony used for FFL client
    :param train_iterator:
    :param val_iterator:
    :param test_iterator:
    :param logger:
    :param local_steps:
    :param tune_locally

    :return:

    """
    if client_type == "mixture":
        return MixtureClient(
            learners_ensemble=learners_ensemble,
            train_iterator=train_iterator,
            val_iterator=val_iterator,
            test_iterator=test_iterator,
            logger=logger,
            local_steps=local_steps,
            tune_locally=tune_locally,
            save_path=save_path
        )
    elif client_type == "AFL":
        return AgnosticFLClient(
            learners_ensemble=learners_ensemble,
            train_iterator=train_iterator,
            val_iterator=val_iterator,
            test_iterator=test_iterator,
            logger=logger,
            local_steps=local_steps,
            tune_locally=tune_locally
        )
    elif client_type == "FFL":
        return FFLClient(
            learners_ensemble=learners_ensemble,
            train_iterator=train_iterator,
            val_iterator=val_iterator,
            test_iterator=test_iterator,
            logger=logger,
            local_steps=local_steps,
            tune_locally=tune_locally,
            q=q
        )
    elif client_type == "Gmixture":
        return GMixtureClient(
            learners_ensemble=learners_ensemble,
            train_iterator=train_iterator,
            val_iterator=val_iterator,
            test_iterator=test_iterator,
            logger=logger,
            local_steps=local_steps,
            tune_locally=tune_locally
        )
    elif client_type == "ACGmixture":
        return ACGMixtureClient(
            learners_ensemble=learners_ensemble,
            train_iterator=train_iterator,
            val_iterator=val_iterator,
            test_iterator=test_iterator,
            logger=logger,
            local_steps=local_steps,
            tune_locally=tune_locally,
            save_path=save_path
        )
    elif client_type == "normal":
        return Client(
            learners_ensemble=learners_ensemble,
            train_iterator=train_iterator,
            val_iterator=val_iterator,
            test_iterator=test_iterator,
            logger=logger,
            local_steps=local_steps,
            tune_locally=tune_locally,
            save_path=save_path
        )
    else:
        raise NotImplementedError(f"{client_type} not recognized client type")


def get_aggregator(
        aggregator_type: object,
        clients: object,
        global_learners_ensemble: object,
        lr: object,
        lr_lambda: object,
        mu: object,
        communication_probability: object,
        q: object,
        sampling_rate: object,
        log_freq: object,
        global_train_logger: object,
        global_test_logger: object,
        test_clients: object,
        verbose: object,
        em_step: object,
        seed: object = None
) -> object:
    """
    `personalized` corresponds to pFedMe

    :param aggregator_type:
    :param clients:
    :param global_learners_ensemble:
    :param lr: oly used with FLL aggregator
    :param lr_lambda: only used with Agnostic aggregator
    :param mu: penalization term, only used with L2SGD
    :param communication_probability: communication probability, only used with L2SGD
    :param q: fairness hyper-parameter, ony used for FFL client
    :param sampling_rate:
    :param log_freq:
    :param global_train_logger:
    :param global_test_logger:
    :param test_clients
    :param verbose: level of verbosity
    :param seed: default is None
    :return:

    """
    seed = (seed if (seed is not None and seed >= 0) else int(time.time()))
    if aggregator_type == "no_communication":
        return NoCommunicationAggregator(
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            log_freq=log_freq,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            test_clients=test_clients,
            sampling_rate=sampling_rate,
            verbose=verbose,
            seed=seed
        )
    elif aggregator_type == "centralized":
        return CentralizedAggregator(
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            log_freq=log_freq,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            test_clients=test_clients,
            sampling_rate=sampling_rate,
            verbose=verbose,
            seed=seed
        )
    elif aggregator_type == "Gcentralized":
        return GCentralizedAggregator(
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            log_freq=log_freq,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            test_clients=test_clients,
            sampling_rate=sampling_rate,
            verbose=verbose,
            seed=seed
        )
    elif aggregator_type == "ACGcentralized":
        return ACGCentralizedAggregator(
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            log_freq=log_freq,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            test_clients=test_clients,
            sampling_rate=sampling_rate,
            verbose=verbose,
            em_step=em_step,
            seed=seed
        )
    elif aggregator_type == "personalized":
        return PersonalizedAggregator(
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            log_freq=log_freq,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            test_clients=test_clients,
            sampling_rate=sampling_rate,
            verbose=verbose,
            seed=seed
        )
    elif aggregator_type == "clustered":
        return ClusteredAggregator(
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            log_freq=log_freq,
            test_clients=test_clients,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            sampling_rate=sampling_rate,
            verbose=verbose,
            seed=seed
        )
    elif aggregator_type == "L2SGD":
        return LoopLessLocalSGDAggregator(
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            log_freq=log_freq,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            test_clients=test_clients,
            communication_probability=communication_probability,
            penalty_parameter=mu,
            sampling_rate=sampling_rate,
            verbose=verbose,
            seed=seed
        )
    elif aggregator_type == "AFL":
        return AgnosticAggregator(
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            log_freq=log_freq,
            test_clients=test_clients,
            lr_lambda=lr_lambda,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            sampling_rate=sampling_rate,
            verbose=verbose,
            seed=seed
        )
    elif aggregator_type == "FFL":
        return FFLAggregator(
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            log_freq=log_freq,
            test_clients=test_clients,
            lr=lr,
            q=q,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            sampling_rate=sampling_rate,
            verbose=verbose,
            seed=seed
        )
    elif aggregator_type == "decentralized":
        n_clients = len(clients)
        mixing_matrix = get_mixing_matrix(n=n_clients, p=0.5, seed=seed)

        return DecentralizedAggregator(
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            mixing_matrix=mixing_matrix,
            log_freq=log_freq,
            test_clients=test_clients,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            sampling_rate=sampling_rate,
            verbose=verbose,
            seed=seed
        )
    else:
        raise NotImplementedError(
            "{aggregator_type} is not a possible aggregator type."
            " Available are: `no_communication`, `centralized`,"
            " `personalized`, `clustered`, `fednova`, `AFL`,"
            " `FFL` and `decentralized`."
        )
