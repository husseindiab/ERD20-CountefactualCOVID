import pandas as pd
import numpy as np
import math
import torch
import torchcde
from torch.nn import functional as F 
from IPython.display import clear_output
import matplotlib.pyplot as plt
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

# We acknowledge the use of tutorials at https://github.com/patrick-kidger/torchcde

class CDEFunc(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels):
        # input_channels is the number of input channels in the data X. (Determined by the data.)
        # hidden_channels is the number of channels for z_t. (Determined by you!)
        super(CDEFunc, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        self.linear1 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.linear2 = torch.nn.Linear(hidden_channels, input_channels * hidden_channels)
        self.elu = torch.nn.ELU(inplace=True)
        self.W = torch.nn.Parameter(torch.Tensor(input_channels))
        self.W.data.fill_(1)
        
    def l2_reg(self):
        '''L2 regularization on all parameters'''
        reg = 0.
        reg += torch.sum(self.linear1.weight ** 2)
        reg += torch.sum(self.linear2.weight ** 2)
        return reg
    
    def l1_reg(self):
        '''L1 regularization on input layer parameters'''
        return torch.sum(torch.abs(self.W))
    
    # The t argument can be ignored or added specifically if you want your CDE to behave differently at
    # different times.
    def forward(self, t, z):
        # z has shape (batch, hidden_channels)
        z = self.linear1(z)
        z = self.elu(z)
        z = self.linear2(z)
        # Ignoring the batch dimension, the shape of the output tensor must be a matrix,
        # because we need it to represent a linear map from R^input_channels to R^hidden_channels.
        z = z.view(z.size(0), self.hidden_channels, self.input_channels)
        z = torch.matmul(z,torch.diag(self.W))
        return z


# Next, we need to package CDEFunc up into a model that computes the integral.
class NeuralCDE(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels):
        super(NeuralCDE, self).__init__()

        self.func = CDEFunc(input_channels, hidden_channels)
        self.initial = torch.nn.Linear(input_channels, hidden_channels)
        self.readout = torch.nn.Linear(hidden_channels, output_channels)

    def forward(self, coeffs):
        X = torchcde.NaturalCubicSpline(coeffs)

        z0 = self.initial(X.evaluate(0.))

        # Actually solve the CDE.
        z_hat = torchcde.cdeint(X=X,
                              z0=z0,
                              func=self.func,
                              t=X.grid_points)
        
        pred_y = self.readout(z_hat.squeeze(0)).unsqueeze(0)
        
        return pred_y
    
    def predict(self, data, z0):
        
        coeffs = torchcde.natural_cubic_spline_coeffs(data)
        X = torchcde.NaturalCubicSpline(coeffs)

        #z0 = X.evaluate(0.)

        # Actually solve the CDE.
        z_hat = torchcde.cdeint(X=X,
                              z0=z0,
                              func=self.func,
                              t=torch.linspace(X.grid_points[0], X.grid_points[-1], 100))

        return z_hat.detach()

    
def predict(model,full_coeffs):
    return model(full_coeffs).detach()


def train(model,train_X, train_y, test_X, test_y, iterations=1000, l1_reg=0.0001, l2_reg=0.001):
    
    optimizer = torch.optim.Adam(model.parameters())
    train_coeffs = torchcde.natural_cubic_spline_coeffs(train_X)

    full_coeffs = torchcde.natural_cubic_spline_coeffs(test_X)

    for i in range(iterations):
        pred_y = model(train_coeffs)
        loss = F.mse_loss(pred_y, train_y)
        loss = loss + l1_reg * model.func.l1_reg() + l2_reg * model.func.l2_reg()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if i % 10 == 0:
            plot_trajectories(test_X, test_y, model=model, full_coeffs=full_coeffs, title=[i,loss])
            clear_output(wait=True)


#=====================================================================================================
# Utilities to load and plot data
#=====================================================================================================

def load_data(feature='death_rate', start_date='2020-3-5', end_date='2020-5-1'):
    """Returns dataframes with selected countries and selected feature"""

    assert feature in ['death_rate', 'reproduction_rate'], 'only features allowed are death_rate and reproduction_rate'

    # select start and end dates
    start_date = pd.to_datetime(start_date).date()
    end_date = pd.to_datetime(end_date).date()

    # select list of countries
    countries = ['AUT', 'BEL', 'DNK', 'FRA', 'DEU', 'GRC', 'ITA', 'NLD', 'NOR', 'PRT', 'ESP', 'SWE']

    # list of countries that will serve as synthetic controls
    df_countries = pd.DataFrame() 

    for c in countries:
        
        # load data and index by date
        df = pd.read_csv(f'./data/processed/owid/owid_{c}.csv')
        df.index = pd.to_datetime(df.date.values).date
        
        # in case of missing values forward-fill with last available
        df = df.fillna(method='pad').dropna(axis=0, how='all').fillna(method='bfill')
        
        # normalize by total population 
        if feature=='death_rate':
            df[feature] = 1e5*(df['new_deaths']/df['population']).ewm(alpha=0.1).mean()
            # df[feature] = (df['new_deaths'].cumsum()/df['population'])
        
        # select only relevant features
        df = df[feature]
        
        # filter to selected dates and append to list
        df_countries[c] = df.loc[start_date:end_date].values.squeeze()

    df_countries.index = df.loc[start_date:end_date].index
    
    return df_countries


def plot_trajectories(X,Y,model,full_coeffs,title=[1,2.1]):
     fig, axs = plt.subplots(1,2, figsize=(10, 2.3))
     fig.tight_layout(pad=0.2, w_pad=2, h_pad=3)
    
     predictions = predict(model,full_coeffs)
     axs[0].plot(Y.squeeze(), label='factual')
     axs[0].plot(predictions.squeeze(), label='counterfactual')
     axs[0].set_title("Iteration = %i" % title[0] + ",  " +  "Loss = %1.3f" % title[1])
     axs[0].legend()
    #  axs[1].plot(Y.squeeze()-predictions.squeeze())
    #  axs[1].set_title('Lockdown effect')
     axs[1].plot(X[0,:,:])
     axs[1].set_title('Control trajectories')
     plt.show()