
# Imports
import gpytorch
import torch
import numpy as np
import matplotlib.pyplot as plt
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.utils import standardize
from gpytorch.mlls import ExactMarginalLogLikelihood

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
    

training_iterations = 100

train_x = torch.from_numpy(X.ravel()).to(dtype=torch.float64)
train_y = torch.from_numpy(Yc.ravel()).to(dtype=torch.float64)



# Our testing script takes in a GPyTorch MLL (objective function) class
# and then trains/tests an approximate GP with it on the supplied dataset

#def train_and_test_approximate_gp(objective_function_cls):
model = ApproximateGPModel(train_x).to(dtype=torch.float64)
#likelihood = gpytorch.likelihoods.GaussianLikelihood()
likelihood = gpytorch.likelihoods.StudentTLikelihood()
objective_function = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y.numel())
# replace with gpytorch mll
optimizer = torch.optim.Adam(list(model.parameters()) + list(likelihood.parameters()), lr=0.1)



# Train
model.train()
#put it into training mode
likelihood.train()

# fits the model 
for _ in range(training_iterations):
    output = model(train_x)
    loss = -objective_function(output, train_y)
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


# fit GP
gp = SingleTaskGP(train_x.reshape(100,1).to(dtype=torch.float64), train_y.reshape(100,1).to(dtype=torch.float64))
mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
fit_gpytorch_mll(mll)

meanGP = gp.posterior(train_x.reshape(100,1).to(dtype=torch.float64)).mean.reshape(100).detach()
stdGP = np.sqrt(gp.posterior(train_x.reshape(100,1).to(dtype=torch.float64)).variance.reshape(100).detach())

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
ax1.plot(train_x, meanSTP, label="lik-t GP")
ax1.fill_between(train_x, meanSTP - 2*stdSTP, meanSTP + 2*stdSTP, color='blue', alpha=0.5)


ax2.plot(train_x, meanGP, label="GP")
ax2.fill_between(train_x, meanGP - 2*stdGP, meanGP + 2*stdGP, color='gray', alpha=0.5)

#ax.plot(train_x, samples)

ax1.scatter(train_x, train_y, c='k', marker='.', label="Data")
ax1.legend(loc="best")
ax1.set(xlabel="x", ylabel="y")
ax1.set_title("Student-t GP")
ax2.scatter(train_x, train_y, c='k', marker='.', label="Data")
ax2.legend(loc="best")
ax2.set(xlabel="x", ylabel="y")
ax2.set_title("Gaussian GP")

plt.show()