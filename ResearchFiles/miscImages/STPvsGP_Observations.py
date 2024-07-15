
# Imports
import gpytorch
import torch
import numpy as np
import matplotlib.pyplot as plt
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.utils import standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from tqdm import trange
import pandas as pd
from gpytorch.likelihoods import GaussianLikelihood

# check for MPS and CUDA device availability
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("MPS device found, setting as device.")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA device found, setting as device.")
else:
    device = torch.device("cpu")
    print("Neither MPS nor CUDA device found. Using default device (CPU).")


# set the seed for all random use
def set_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seeds(22)

# Create the training data
real_std = 0.1
X = np.linspace(0.0, np.pi * 2, 100)[:, None]
Y = np.sin(X) + np.random.randn(*X.shape) * real_std
Y = Y / Y.max()
Yc = Y.copy()
X = X/X.max()
# Slightly noisy data
Yc[75:80] += 1



class ApproximateGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(-1))
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=False
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class GPModel(gpytorch.models.ExactGP):
    def __init__(self, x_train, y_train, likelihood):
        # build the model using the ExactGP model from gpytorch
        super(GPModel, self).__init__(x_train, y_train, likelihood)

        # use a constant mean, this value can be learned from the dataset
        self.mean_module = gpytorch.means.ConstantMean()

        # automatically determine the number of dimensions for the ARD kernel
        num_dimensions = x_train.dim()

        # use a scaled Matern kernel, the ScaleKernel allows the kernel to learn a scale factor for the dataset
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5,ard_num_dims=num_dimensions))
            
        # set the number of outputs 
        self.num_outputs = 1

    def forward(self, x):
        # forward pass of the model

        # compute the mean and covariance of the model 
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)

        # return the MultivariateNormal distribution of the mean and covariance 
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    
def UCB(mean, std, beta):
    return mean + beta * std
# Initial setup
training_iterations = 20
num_initial_points = 10
num_new_samples_per_iteration = 1  # Change this to your desired number of samples per iteration
total_samples = 100


# Randomly initialize training data with 10 points
train_x = torch.from_numpy(X.ravel()).to(dtype=torch.float64)
train_y = torch.from_numpy(Yc.ravel()).to(dtype=torch.float64)

indices = torch.randperm(total_samples)[:num_initial_points]
train_xSTP = train_x[indices]
train_ySTP = train_y[indices]
train_xGP = train_xSTP
train_yGP = train_ySTP
initialPoints = train_xSTP
initialPointsy = train_ySTP

# Our testing script takes in a GPyTorch MLL (objective function) class
# and then trains/tests an approximate GP with it on the supplied dataset

# fits the model 
for _ in trange(training_iterations):
    #def train_and_test_approximate_gp(objective_function_cls):
    model = ApproximateGPModel(train_xSTP).to(dtype=torch.float64)
    #likelihood = gpytorch.likelihoods.GaussianLikelihood()
    likelihood = gpytorch.likelihoods.StudentTLikelihood()
    objective_function = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_ySTP.numel())
    # replace with gpytorch mll
    optimizer = torch.optim.Adam(list(model.parameters()) + list(likelihood.parameters()), lr=0.1)


    # Train
    model.train()
    #put it into training mode
    likelihood.train()

    for _ in range(50):
        output = model(train_xSTP)
        loss = -objective_function(output, train_ySTP)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


    # Test
    model.eval()
    likelihood.eval()

    # works wtih the variational distribution
    with gpytorch.settings.num_likelihood_samples(512):
        observed_pred = likelihood(model(train_x))

    samples = observed_pred.sample()


    # use as inputs to UCB 
    meanSTP = samples.mean(dim=0)
    stdSTP = samples.std(dim=0)
    

    # Select new points using UCB
    ucb_values = UCB(meanSTP, stdSTP, 1.96)
    ucb_values[indices] = -float('inf')
    indices = torch.cat([indices, ucb_values.argmax().unsqueeze(0)])


    # Convert new_indices to a tensor
    # Add the new points to the training data
    train_xSTP = train_x[indices]
    train_ySTP = train_y[indices]
    
indices = torch.randperm(total_samples)[:num_initial_points]

# optimize the model
for i in trange(training_iterations):

    bestY = train_yGP.max().item()

    # optimize the model
    # use a half normal prior for the noise to find a Gaussian likelihood
    likelihood = GaussianLikelihood(noise_prior=gpytorch.priors.HalfNormalPrior(0.01))

    # using the found likelihood, create a GP model
    gp = GPModel(train_xGP, train_yGP, likelihood)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp)

    # fit the model by maximizing the marginal log likelihood
    gp.train()
    likelihood.train()
    fit_gpytorch_mll(mll)

    gp.eval()
    likelihood.eval()

    # predict from candidate pool
    with torch.no_grad():
        pred = gp(train_x) # predict values for all candidates

    meanGP = pred.mean
    stdGP = pred.stddev

    # pass the predictions through an acquisition function to find the next best point to sample
    acqVal = UCB(meanGP, stdGP, 1.96)
    acqVal[indices] = -float('inf')  # Don't select already sampled points
    indices = torch.cat([indices, acqVal.argmax().unsqueeze(0)]) # add best value to index

    # add the new point to the training data
    train_xGP = train_x[indices]
    train_yGP = train_y[indices]

# Create subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))  # Adjusted figsize for better visibility

# First subplot
ax1.fill_between(train_x.numpy(), (meanSTP - 2*stdSTP).numpy(), (meanSTP + 2*stdSTP).numpy(), color='grey', alpha=0.5, label='Uncertainty (lik-t GP)')
ax1.scatter(train_xSTP.numpy(), train_ySTP.numpy(), c='red', label="Student-T Process")
ax1.plot(train_x.numpy(), meanSTP.numpy(), 'b-', label="lik-t mean")
ax1.legend(loc="best")
ax1.set(xlabel="x", ylabel="y", title="Student-T Process")
ax1.scatter(train_x.numpy(), train_y.numpy(), c='k', marker='.', label="Data")
ax1.scatter(initialPoints, initialPointsy, c='green', marker='o', label="Initial Points")

# Second subplot
ax2.fill_between(train_x.numpy(), (meanGP - 2*stdGP).numpy(), (meanGP + 2*stdGP).numpy(), color='grey', alpha=0.5, label='Uncertainty (GP)')
ax2.scatter(train_xGP.numpy(), train_yGP.numpy(), c='red', label="Observed Points")
ax2.plot(train_x.numpy(), meanGP.numpy(), 'b-', label="GP mean")
ax2.scatter(train_x.numpy(), train_y.numpy(), c='k', marker='.', label="Data")
ax2.scatter(initialPoints, initialPointsy, c='green', marker='o', label="Initial Points")
ax2.legend(loc="best")
ax2.set(xlabel="x", ylabel="y", title="GP")

# Adjust layout
plt.tight_layout()
plt.show()