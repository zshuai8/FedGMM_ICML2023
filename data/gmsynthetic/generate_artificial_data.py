"""
Generates synthetic dataset following a mixture model.
"""
import argparse
import os
import pickle

import numpy as np
from sklearn.utils import shuffle
import scipy.stats as stats
from scipy.stats import laplace


BOX = (-1.0, 1.0)
PATH = "all_data/"


class SyntheticMixtureDataset:
    def __init__(
            self,
            n_classes,
            n_components,
            n_tasks,
            dim,
            noise_level,
            alpha,
            box,
            marginal,
            seed
    ):

        if n_classes != 2:
            raise NotImplementedError("Only binary classification is supported for the moment")
        import numpy as np
        self.n_classes = n_classes
        self.n_components = n_components
        self.n_tasks = n_tasks
        self.dim = dim
        self.noise_level = noise_level
        self.alpha = alpha * np.ones(n_components)
        self.box = box
        self.marginal = marginal
        self.seed = seed
        self.beta =[]

        np.random.seed(self.seed)
        self.num_samples = get_num_samples(self.n_tasks)

        self.theta = np.zeros((self.n_components, self.dim))
        self.mu = np.zeros((self.n_components, self.dim))
        self.var = np.tile(np.eye(self.dim), [self.n_components, 1, 1])
        self.mixture_weights = np.zeros((self.n_tasks, self.n_components))
        
        self.theta2 = np.zeros((self.n_components, self.dim))
        self.mu2 = np.zeros((self.n_components, self.dim))
        self.var2 = np.tile(np.eye(self.dim), [self.n_components, 1, 1])
        self.mixture_weights2 = np.zeros((self.n_tasks, self.n_components))
        import numpy as np

        self.generate_mixture_weights()
        
        self.generate_components()
        self.generate_components_extra()

    def generate_mixture_weights(self):
        for task_id in range(self.n_tasks):
            self.mixture_weights[task_id] = np.random.dirichlet(alpha=self.alpha)
    
    def generate_components(self):
        self.mu = np.random.uniform(BOX[0], BOX[1], size=(self.n_components, self.dim))
        self.var = (1 / np.sqrt(self.dim)) * self.var

        for i in range(self.n_components):

            start_mu = self.mu[i]
            rd_idx = np.random.randint(self.n_components - 1)
            end_mu = np.delete(self.mu, i, 0)[rd_idx]
            rad = 2 * np.random.binomial(1, 0.5, 1)[0] - 1
            self.theta[i] = rad * (end_mu - start_mu)

    def generate_components_extra(self):
        if self.marginal == 'poisson':
            self.mu2 = np.random.uniform(0, BOX[1], size=(self.n_components, self.dim))
        else:
            self.mu2 = np.random.uniform(BOX[0], BOX[1], size=(self.n_components, self.dim))
        self.var2 = (1 / np.sqrt(self.dim)) * self.var

        for i in range(self.n_components):
            if self.marginal == 'beta':
                alpha_dist = np.array([stats.gamma(a=3, scale=1).rvs() for i in range(self.dim)])
                beta_dist = np.array([stats.gamma(a=5, scale=1).rvs() for i in range(self.dim)])
                self.beta.append((alpha_dist, beta_dist))
                self.mu2[i] = alpha_dist / (alpha_dist + beta_dist)
            start_mu = self.mu2[i]
            rd_idx = np.random.randint(self.n_components - 1)
            end_mu = np.delete(self.mu2, i, 0)[rd_idx]
            rad = 2 * np.random.binomial(1, 0.5, 1)[0] - 1
            self.theta2[i] = rad * (end_mu - start_mu)
            if self.marginal == 'beta':
                alpha_dist = [stats.gamma(a=3, scale=1).rvs() for i in range(self.dim)]
                beta_dist = [stats.gamma(a=5, scale=1).rvs() for i in range(self.dim)]
                self.beta.append((alpha_dist, beta_dist))
                
    

    def generate_data(self, task_id, n_samples=10000, mis=False):
        latent_variable_count = np.random.multinomial(n_samples, self.mixture_weights[task_id])
        y = np.zeros(n_samples)

        if self.marginal == "uniform":
            x = np.random.uniform(self.box[0], self.box[1], size=(n_samples, self.dim))
        elif self.marginal == "gaussian":
            x = np.zeros([n_samples, self.dim])
            # x will be generated later along with y
        elif self.marginal == 'poisson':

            x = np.zeros([n_samples, self.dim])
        elif self.marginal == 'beta':

            x = np.zeros([n_samples, self.dim])
        elif self.marginal == 'laplace':

            x = np.zeros([n_samples, self.dim])
        else:
            
            raise NotImplementedError("Only uniform marginal is available for the moment")

        current_index = 0
        for component_id in range(self.n_components):
            n_sample = latent_variable_count[component_id]
            print(current_index, current_index + n_sample)
            """
               change distribution here
               1) change distribution 
               2) 100 client 3 base distribution, last 10 client different mu and change distribution, y hat distribution change
            """
            if not mis:
                x[current_index:current_index + n_sample] = \
                    np.random.multivariate_normal(mean=self.mu[component_id],
                                                  cov=self.var[component_id], size=n_sample)
                y_hat = sigmoid((x[current_index:current_index + n_sample]
                             - self.mu[component_id]) @ self.theta[component_id])

                y[current_index: current_index + n_sample] = \
                    np.round(y_hat).astype(int)
            else:
                if self.marginal == 'poisson':
                    poisson_mean = self.mu2[component_id]
                    x = np.random.poisson(poisson_mean, size=(n_samples, self.dim))
                elif self.marginal == 'beta':
                    # define the gamma distributions for Alpha and Beta
                    alpha_dist = self.beta[component_id][0]
                    beta_dist = self.beta[component_id][1]
                    for i in range(self.dim):
                        x[current_index:current_index + n_sample,i] = np.random.beta(alpha_dist[i], beta_dist[i], size=n_sample)
                elif self.marginal == 'laplace':
                    for i in range(self.dim):
                        x[current_index:current_index + n_sample,i] = \
                        np.random.laplace(loc=self.mu2[component_id][i],scale=1/np.sqrt(self.dim), size=n_sample)
                y_hat = sigmoid((x[current_index:current_index + n_sample]
                             - self.mu2[component_id]) @ self.theta2[component_id])

                y[current_index: current_index + n_sample] = \
                    np.round(y_hat).astype(int)
            

            current_index = current_index + n_sample

        return shuffle(x, y)

    def save_metadata(self, path_):
        metadata = dict()
        metadata["mixture_weights"] = self.mixture_weights
        metadata["theta"] = self.theta
        metadata["mu"] = self.mu
        metadata["var"] = self.var

        print(self.mu)
        print(self.theta)

        with open(path_, 'wb') as f:
            pickle.dump(metadata, f)


def save_data(x, y, path_):
    data = list(zip(x, y))
    with open(path_, 'wb') as f:
        pickle.dump(data, f)


def get_num_samples(num_tasks, min_num_samples=100, max_num_samples=1000):
    num_samples = np.random.lognormal(4, 2, num_tasks).astype(int)
    num_samples = [min(s + min_num_samples, max_num_samples) for s in num_samples]
    return num_samples


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    ex = np.exp(x)
    sum_ex = np.sum(np.exp(x))
    return ex / sum_ex


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--n_tasks',
        help='number of tasks/clients;',
        required=True,
        type=int
    )
    parser.add_argument(
        '--n_classes',
        help='number of classes;',
        type=int,
        default=2
    )
    parser.add_argument(
        '--n_components',
        help='number of classes;',
        type=int,
        default=3
    )
    parser.add_argument(
        '--dimension',
        help='data dimension;',
        type=int,
        default=150
    )
    parser.add_argument(
        '--noise_level',
        help='Noise level;',
        type=float,
        default=0.1
    )
    parser.add_argument(
        '--n_test',
        help='size of test set;',
        type=int,
        default=5_000
    )
    parser.add_argument(
        '--train_tasks_frac',
        help='fraction of tasks / clients  participating to the training; default is 0.0',
        type=float,
        default=1.0
    )
    parser.add_argument(
        '--alpha',
        help='parameter controlling tasks dissimilarity, the smaller alpha is the more tasks are dissimilar;',
        type=float,
        default=0.4
    )
    parser.add_argument(
        '--marginal',
        help='a string indicating the shape of the marginal for x (uniform | gaussian);',
        type=str,
        default="beta"
    )
    parser.add_argument(
        '--seed',
        help='seed for the random processes;',
        type=int,
        default=12345,
        required=False
    )
    return parser.parse_args()


def main():
    os.makedirs(PATH, exist_ok=True)
    args = parse_args()

    np.random.seed(args.seed)
    dataset = \
        SyntheticMixtureDataset(
            n_components=args.n_components,
            n_classes=args.n_classes,
            n_tasks=args.n_tasks,
            dim=args.dimension,
            noise_level=args.noise_level,
            alpha=args.alpha,
            marginal=args.marginal,
            seed=args.seed,
            box=BOX,
        )

    dataset.save_metadata(os.path.join(PATH, "meta.pkl"))

    os.makedirs(os.path.join(PATH, "train"), exist_ok=True)
    os.makedirs(os.path.join(PATH, "test"), exist_ok=True)

    for task_id in range(dataset.n_tasks-10):
        if task_id < int(args.train_tasks_frac * args.n_tasks):
            mode = "train"
        else:
            mode = "test"

        client_path = os.path.join(PATH, mode, "task_{}".format(task_id))
        os.makedirs(client_path, exist_ok=True)

        x_train, y_train = dataset.generate_data(task_id, dataset.num_samples[task_id])
        x_test, y_test = dataset.generate_data(task_id, args.n_test)

        save_data(x_train, y_train, os.path.join(client_path, "train.pkl"))
        save_data(x_test, y_test, os.path.join(client_path, "test.pkl"))
    for task_id in range(dataset.n_tasks-10, dataset.n_tasks):
        if task_id < int(args.train_tasks_frac * args.n_tasks):
            mode = "train"
        else:
            mode = "test"

        client_path = os.path.join(PATH, mode, "task_{}".format(task_id))
        os.makedirs(client_path, exist_ok=True)

        x_train, y_train = dataset.generate_data(task_id, dataset.num_samples[task_id], mis=True)
        x_test, y_test = dataset.generate_data(task_id, args.n_test, mis=True)

        save_data(x_train, y_train, os.path.join(client_path, "train.pkl"))
        save_data(x_test, y_test, os.path.join(client_path, "test.pkl"))


if __name__ == '__main__':
    main()
