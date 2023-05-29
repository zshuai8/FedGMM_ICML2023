# Personalized Federated Learning under Mixture of Distributions

This repository is the official implementation of [Personalized Federated Learning under Mixture of Distributions](https://arxiv.org/abs/2305.01068).

The recent trend towards Personalized Federated Learning (PFL) has garnered significant attention as it allows for the training of models that are tailored to each client while maintaining data privacy. However, current PFL techniques primarily focus on modeling the conditional distribution heterogeneity (i.e. concept shift), which can result in suboptimal performance when the distribution of input data across clients diverges (i.e. covariate shift). Additionally, these techniques often lack the ability to adapt to unseen data, further limiting their effectiveness in real-world scenarios.

To address these limitations, we propose a novel approach, FedGMM, which utilizes Gaussian mixture models (GMM) to effectively fit the input data distributions across diverse clients. The model parameters are estimated by maximum likelihood estimation utilizing a federated Expectation-Maximization algorithm, which is solved in closed form and does not assume gradient similarity. Furthermore, \ourmethod\ possesses an additional advantage of adapting to new clients with minimal overhead, and it also enables uncertainty quantification. Empirical evaluations on synthetic and benchmark datasets demonstrate the superior performance of our method in both PFL classification and novel sample detection.

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Usage

We provide code to simulate federated training of machine learning. 
The core objects are `Aggregator` and `Client`, different federated learning
algorithms can be implemented by revising the local update method 
`Client.step()` and/or the aggregation protocol defined in
`Aggregator.mix()` and `Aggregator.update_client()`.

In addition to the trivial baseline consisting of training models locally without
any collaboration, this repository supports the following federated learning
algorithms:

* FedEM. ([Marfoq et al. 2021](https://proceedings.neurips.cc/paper/2021/file/82599a4ec94aca066873c99b4c741ed8-Paper.pdf))
* FedAvg ([McMahan et al. 2017](http://proceedings.mlr.press/v54/mcmahan17a.html))
* FdProx ([Li et al. 2018](https://arxiv.org/abs/1812.06127))
* Clustered FL ([Sattler et al. 2019](https://ieeexplore.ieee.org/abstract/document/9174890))
* pFedMe ([Dinh et al. 2020](https://proceedings.neurips.cc/paper/2020/file/f4f1f13c8289ac1b1ee0ff176b56fc60-Paper.pdf))
* L2SGD ([Hanzely et al. 2020](https://proceedings.neurips.cc/paper/2020/file/187acf7982f3c169b3075132380986e4-Paper.pdf))
* APFL ([Deng et al. 2020](https://arxiv.org/abs/2003.13461))
* q-FFL ([Li et al. 2020](https://openreview.net/forum?id=ByexElSYDr))
* AFL ([Mohri et al. 2019](http://proceedings.mlr.press/v97/mohri19a.html))

## Datasets

We provide five federated benchmark datasets spanning a wide range
of machine learning tasks: image classification (CIFAR10 and CIFAR100),
handwritten character recognition (EMNIST and FEMNIST), and language
modelling (Shakespeare), in addition to a synthetic dataset

Shakespeare dataset (resp. FEMNIST) was naturally partitioned by assigning
all lines from the same characters (resp. all images from the same writer)
to the same client.  We created federated versions of CIFAR10 and EMNIST by
distributing samples with the same label across the clients according to a 
symmetric Dirichlet distribution with parameter 0.4. For CIFAR100,
we exploited the availability of "coarse" and "fine" labels, using a two-stage
Pachinko allocation method  to assign 600 sample to each of the 100 clients.

The following table summarizes the datasets and models

|Dataset         | Task |  Model |
| ------------------  |  ------|------- |
| FEMNIST   |     Handwritten character recognition       |     2-layer CNN + 2-layer FFN  |
| EMNIST    |    Handwritten character recognition     |      2-layer CNN + 2-layer FFN     |
| CIFAR10   |     Image classification        |      MobileNet-v2 |
| CIFAR100    |     Image classification         |      MobileNet-v2  |
| Shakespeare |     Next character prediction        |      Stacked LSTM    |
| Synthetic dataset| Binary classification | Linear model | 

See the `README.md` files of respective dataset, i.e., `data/$DATASET`,
for instructions on generating data

## Training

Run on one dataset, with a specific  choice of federated learning method.
Specify the name of the dataset (experiment), the used method, and configure all other
hyper-parameters (see all hyper-parameters values in the appendix of the paper)

```train
python3  python run_experiment.py cifar10 FedGMM \
    --n_learners 3 \
    --n_gmm 3
    --n_rounds 200 \
    --bz 128 \
    --lr 0.01 \
    --lr_scheduler multi_step \
    --log_freq 5 \
    --device cuda \
    --optimizer sgd \
    --seed 1234 \
    --logs_root ./logs \
    --verbose 1
```

The test and training accuracy and loss will be saved in the specified log path.

We provide example scripts to run paper experiments under `scripts/` directory.

## Evaluation

We give instructions to run experiments on CIFAR-10 dataset as an example
(the same holds for the other datasets). You need first to go to 
`./data/cifar10`, follow the instructions in `README.md` to download and partition
the dataset.

All experiments will generate tensorboard log files (`logs/cifar10`) that you can 
interact with, using [TensorBoard](https://www.tensorflow.org/tensorboard)

### Average performance of personalized models

Run the following scripts, this will generate tensorboard logs that you can interact with to make plots or get the
values presented in Table 2

```eval
# run FedAvg
echo "Run FedAvg"
python run_experiment.py cifar10 FedAvg --n_learners 1 --n_rounds 200 --bz 128 --lr 0.01 \
 --lr_scheduler multi_step --log_freq 5 --device cuda --optimizer sgd --seed 1234 --verbose 1

# run FedAvg + local adaption
echo "run FedAvg + local adaption"
python run_experiment.py cifar10 FedAvg --n_learners 1 --locally_tune_clients --n_rounds 201 --bz 128 \
 --lr 0.001 --lr_scheduler multi_step --log_freq 10 --device cuda --optimizer sgd --seed 1234 --verbose 1

# run training using local data only
echo "Run Local"
python run_experiment.py cifar10 local --n_learners 1 --n_rounds 201 --bz 128 --lr 0.03 \
 --lr_scheduler multi_step --log_freq 10 --device cuda --optimizer sgd --seed 1234 --verbose 1

# run Clustered FL
echo "Run Clustered FL"
python run_experiment.py cifar10 clustered --n_learners 1 --n_rounds 201 --bz 128 --lr 0.003 \
 --lr_scheduler multi_step --log_freq 10 --device cuda --optimizer sgd --seed 1234 --verbose 1

# run FedProx
echo "Run FedProx"
python run_experiment.py cifar10 FedProx --n_learners 1 --n_rounds 201 --bz 128 --lr 0.01 --mu 1.0\
 --lr_scheduler multi_step --log_freq 10 --device cuda --optimizer prox_sgd --seed 1234 --verbose 1

# Run pFedME
echo "Run "
python run_experiment.py cifar10 pFedMe --n_learners 1 --n_rounds 201 --bz 128 --lr 0.001 --mu 1.0 \
 --lr_scheduler multi_step --log_freq 10 --device cuda --optimizer prox_sgd --seed 1234 --verbose 1

# run FedEM
echo "Run FedEM"
python run_experiment.py cifar10 FedEM --n_learners 3 --n_rounds 201 --bz 128 --lr 0.03 \
 --lr_scheduler multi_step --log_freq 10 --device cuda --optimizer sgd --seed 1234 --verbose 1
 
# run FedGMM
echo "Run FedEM"
python run_experiment.py cifar10 FedGMM --n_learners 3 --n_gmm 3 --n_rounds 201 --bz 128 --lr 0.03 \
 --lr_scheduler multi_step --log_freq 10 --device cuda --optimizer sgd --seed 1234 --verbose 1
```

Similar for other datasets are provided in `papers_experiments/`


