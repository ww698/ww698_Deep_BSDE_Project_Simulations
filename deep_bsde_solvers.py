
# Credit: adapted from https://github.com/YifanJiang233/Deep_BSDE_solver/tree/master

# Implemented with analytical Jacobian for sigma for faster computation

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class fbsde():
    def __init__(self, x_0, b, sigma, dsigma, f, g, T, dim_x,dim_y,dim_d):
        self.x_0 = x_0.to(device)
        self.b = b
        self.sigma = sigma
        self.f = f
        self.g = g
        self.T = T
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_d = 1 # we stick to case d=1
        self.dsigma = dsigma # Additional Jacobian term (batch_size, dim_x, dim_x)


# Deep BSDE with Milstein for Forwards SDE for case dim_d = 1

class Model_Milstein(nn.Module):
    def __init__(self, equation, dim_h):
        super(Model_Milstein, self).__init__()
        self.linear1 = nn.Linear(equation.dim_x+1, dim_h)
        self.linear2 = nn.Linear(dim_h, dim_h)
        self.linear3 = nn.Linear(dim_h, equation.dim_y*equation.dim_d)
        self.y_0 = nn.Parameter(torch.rand(equation.dim_y, device=device))
        
        self.equation= equation


    def forward(self,batch_size, N):
        def phi(x):
            x = F.relu(self.linear1(x))
            x = F.relu(self.linear2(x))
            return self.linear3(x).reshape(-1, self.equation.dim_y, self.equation.dim_d)

        delta_t = self.equation.T / N
        
        W = torch.randn(batch_size, self.equation.dim_d, N, device=device) * np.sqrt(delta_t)
           # N brownian increments for each batch
        x = self.equation.x_0+torch.zeros(W.size()[0],self.equation.dim_x,device=device)
        y = self.y_0+torch.zeros(W.size()[0],self.equation.dim_y,device=device)
        for i in range(N):
            u = torch.cat((x, torch.ones(x.size()[0], 1,device=device)*delta_t*i), 1)
            z = phi(u)
            w = W[:, :, i].reshape(-1, self.equation.dim_d, 1) # [batch, 1]
            
            sigma_milstein = torch.einsum('brk,bk->br',self.equation.dsigma(delta_t*i, x),
                               self.equation.sigma(delta_t*i, x).squeeze(-1))
            milstein  = 0.5 * sigma_milstein * (w.squeeze(-1)**2 - delta_t)
            x = (
                x+self.equation.b(delta_t*i, x, y)*delta_t
                +torch.matmul( self.equation.sigma(delta_t*i, x), w).reshape(-1, self.equation.dim_x)
                + milstein
            )# added milstein term
            
            y = y-self.equation.f(delta_t*i, x, y, z)*delta_t + torch.matmul(z, w).reshape(-1, self.equation.dim_y)
        return x, y


class BSDEsolver_Milstein():
    def __init__(self, equation, dim_h):
        self.model = Model_Milstein(equation,dim_h).to(device)
        self.equation = equation

    def train(self, batch_size, N, itr):
        criterion = torch.nn.MSELoss().to(device)

        optimizer = torch.optim.Adam(self.model.parameters())

        loss_data, y0_data = [], []

        for i in range(itr):
            x, y = self.model(batch_size,N)
            loss = criterion(self.equation.g(x), y)
            loss_data.append(float(loss))
            y0_data.append(float(self.model.y_0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return loss_data, y0_data

# E-M Solvers, can be used for general dim_x and dim_d

class Model_Euler(nn.Module):
    def __init__(self, equation, dim_h):
        super(Model_Euler, self).__init__()
        self.linear1 = nn.Linear(equation.dim_x+1, dim_h)
        self.linear2 = nn.Linear(dim_h, dim_h)
        self.linear3 = nn.Linear(dim_h, equation.dim_y*equation.dim_d)
        self.y_0 = nn.Parameter(torch.rand(equation.dim_y, device=device))
        
        self.equation= equation


    def forward(self,batch_size, N):
        def phi(x):
            x = F.relu(self.linear1(x))
            x = F.relu(self.linear2(x))
            return self.linear3(x).reshape(-1, self.equation.dim_y, self.equation.dim_d)

        delta_t = self.equation.T / N
        
        W = torch.randn(batch_size, self.equation.dim_d, N, device=device) * np.sqrt(delta_t)
       
        x = self.equation.x_0+torch.zeros(W.size()[0],self.equation.dim_x,device=device)
        y = self.y_0+torch.zeros(W.size()[0],self.equation.dim_y,device=device)
        for i in range(N):
            u = torch.cat((x, torch.ones(x.size()[0], 1,device=device)*delta_t*i), 1)
            z = phi(u)
            w = W[:, :, i].reshape(-1, self.equation.dim_d, 1)
            x = x+self.equation.b(delta_t*i, x, y)*delta_t+torch.matmul( self.equation.sigma(delta_t*i, x), w).reshape(-1, self.equation.dim_x)
            y = y-self.equation.f(delta_t*i, x, y, z)*delta_t + torch.matmul(z, w).reshape(-1, self.equation.dim_y)
        return x, y


class BSDEsolver_Euler():
    def __init__(self, equation, dim_h):
        self.model = Model_Euler(equation,dim_h).to(device)
        self.equation = equation

    def train(self, batch_size, N, itr):
        criterion = torch.nn.MSELoss().to(device)

        optimizer = torch.optim.Adam(self.model.parameters())

        loss_data, y0_data = [], []

        for i in range(itr):
            x, y = self.model(batch_size,N)
            loss = criterion(self.equation.g(x), y)
            loss_data.append(float(loss))
            y0_data.append(float(self.model.y_0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return loss_data, y0_data
