import torch
import numpy as np
from tqdm import tqdm
import pickle
import sys

from utilities import MLP, loss_function_Z, loss_function_Y

class EulerBSDE:
    def __init__(self, pi, d, T, b, sigma, g, f, device):
        """
        pi: list with the grid at which we want to peform ethe Euler scheme
        d: dimension of the Brownian motion
        T: time horizon

        Let N be the number of samples for the Monte Carlo.

        (t,x,y,z), where
                t is a float,
                x: tensor of size (N, m), 
                y is a tensor of size (N), 
                z is a tensor of size (N, d)
        
        b:      drift of the FSDE. Input (t,x), Output: tensor of size (N, m).
        sigma:  volatility of the FSDE. Input (t,x), Output: tensor of size (N, m, d)
        g:      final condition. Input (x), Ouput: tensor if size (N)
        f:      vector field of the BSDE. Input (t,x,y,z), Output: tensor of size (N).
                
        """
        self.d, self.T, self.pi, self.b, self.sigma, self.g, self.f = d, T, pi, b, sigma, g, f
        self.len_pi = len(pi)

        # arch_Y, lr_Y and act_Y are dictionaries such that arch_Y[k], lr_Y[k] and act_Y[k]
        # indicate the architecture of the neural network that parameterizes Y_tk, the initial learning rate that was used and 
        # the activation function. Same for Z.
        self.arch_Y, self.lr_Y, self.act_Y = {}, {}, {}
        self.arch_Z, self.lr_Z, self.act_Z = {}, {}, {}
        self.device = device

    def set_tk(self, k):
        """
        This function sets up all the necessary parameters before beggining to train the networks for the step tk.
        """
        self.k = k
        self.tk, self.tk_1 = self.pi[k], self.pi[k+1]
        self.Delta_tk = self.pi[k+1] - self.pi[k]
        self.wienerchaos.set_tm([self.tk, self.tk_1])
        if k>0:
            self.dim_X[k] = len(self.wienerchaos.index_m[0])
        elif k == 0:
            self.dim_X[k] = 0
        
    def initialize_ZNN_tk(self, arch, act, lr):
        """
        This function initializes the network for Z in step t_k.
        """
        self.arch_Z[self.k], self.lr_Z[self.k], self.act_Z[self.k] = arch, lr, act
        self.Z_NN_tk = MLP(self.dim_X[self.k], self.d, arch, act).to(self.device)
        self.opt_Z_tk = torch.optim.Adam(self.Z_NN_tk.parameters(), lr=lr)
        n_params = "{:,}".format(sum(p.numel() for p in self.Z_NN_tk.parameters()))
        print(f"Number of parameters Z_t{self.k}: {n_params}")

    def initialize_YNN_tk(self, arch, act, lr):
        """
        This function initializes the network for Y in step t_k.
        """
        self.arch_Y[self.k], self.lr_Y[self.k], self.act_Y[self.k] = arch, lr, act
        self.Y_NN_tk = MLP(self.dim_X[self.k], 1, arch, act).to(self.device)
        self.opt_Y_tk = torch.optim.Adam(self.Y_NN_tk.parameters(), lr=lr)
        n_params = "{:,}".format(sum(p.numel() for p in self.Y_NN_tk.parameters()))
        print(f"Number of parameters Y_t{self.k}: {n_params}")

    def load_ZNN_tk(self, path):
        """
        This function loads the network's parameters for Z in step t_k, necessary for the training of YNN_tk
        """
        # We load the dictionary with the hyperparameters
        with open('Models/' + path + '/Hyperparameters/arch_Z.pickle', 'rb') as f1:
            arch_Z = pickle.load(f1)
        with open('Models/' + path + '/Hyperparameters/act_Z.pickle', 'rb') as f2:
            act_Z = pickle.load(f2)
        arch, act = arch_Z[self.k], act_Z[self.k]
        
        if self.k > 0:
            self.Z_NN_tk = MLP(self.dim_X[self.k], self.d, arch, act).to(self.device)
            self.Z_NN_tk.load_state_dict(torch.load('Models/' + path + '/Parameters/Z_NN_t_' + str(self.k) +'.pth'))
        else:
            self.Z0 = torch.load('Models/' + path + '/Parameters/Z0.pt')

    def load_YNN_tk_1(self, path):
        """
        This function loads the network's parameters for Y in step t_k+1, necessary for the training of Z_tk
        """
        if self.tk == self.pi[-2]:
            self.Y_NN_tk_1 = None
        else:
            # We load the dictionary with the hyperparameters
            with open('Models/' + path + '/Hyperparameters/arch_Y.pickle', 'rb') as f1:
                arch_Y = pickle.load(f1)
            with open('Models/' + path + '/Hyperparameters/act_Y.pickle', 'rb') as f2:
                act_Y = pickle.load(f2)
            arch, act = arch_Y[self.k+1], act_Y[self.k+1]
                
            self.Y_NN_tk_1 = MLP(self.dim_X[self.k+1], 1, arch, act).to(self.device)
            self.Y_NN_tk_1.load_state_dict(torch.load('Models/' + path + '/Parameters/Y_NN_t_' + str(self.k+1) +'.pth'))

    def save_ZNN_tk(self, path):
        with open('Models/' + path + '/Hyperparameters/arch_Z.pickle', 'wb') as f:
            pickle.dump(self.arch_Z, f)
        with open('Models/' + path + '/Hyperparameters/act_Z.pickle', 'wb') as f:
            pickle.dump(self.act_Z, f)
        if self.k > 0:
            torch.save(self.Z_NN_tk.state_dict(), 'Models/' + path + '/Parameters/Z_NN_t_' + str(self.k) +'.pth')
        else:
            torch.save(self.Z0, 'Models/' + path + '/Parameters/Z0.pt')

    def save_YNN_tk(self, path):
        with open('Models/' + path + '/Hyperparameters/arch_Y.pickle', 'wb') as f:
            pickle.dump(self.arch_Y, f)
        with open('Models/' + path + '/Hyperparameters/act_Y.pickle', 'wb') as f:
            pickle.dump(self.act_Y, f)
        if self.k > 0:
            torch.save(self.Y_NN_tk.state_dict(), 'Models/' + path + '/Parameters/Y_NN_t_' + str(self.k) +'.pth')
        else:
            torch.save(self.Y0, 'Models/' + path + '/Parameters/Y0.pt')

    def sample_X_tk(self, N):
        """
        This function samples X_tk, X_tk_1 and DeltaBm_tk := Bm_tk_1-Bm_tk, which are necessary to train the networks for 
        Y_tk and Z_tk.
        """
        X, DeltaBm_tk, bm = self.wienerchaos.sample_X_tm(N, [self.pi[self.k], self.pi[self.k+1]], self.wienerchaos.Lambda_n01m)
        X_tk, X_tk_1 = X[0], X[1]
        return X_tk, X_tk_1, DeltaBm_tk, bm

    def sample_for_training_Z_tk(self, N):
        X_tk, X_tk_1, DeltaBm_tk, bm = self.sample_X_tk(N) 
        if self.tk == self.pi[-2]:
            Y_tk_1 = torch.sum(self.chaos_coeffs.unsqueeze(0) * X_tk_1, dim=1).unsqueeze(1).unsqueeze(1)
        else:
            with torch.no_grad():
                Y_tk_1 = self.Y_NN_tk_1(X_tk_1)  
    
        if self.k > 0:
            Z_tk = self.Z_NN_tk(X_tk)
        else:
            Z_tk = None
         
        return Z_tk, Y_tk_1, DeltaBm_tk

    def sample_for_training_Y_tk(self, N):
        X_tk, X_tk_1, DeltaBm_tk, bm = self.sample_X_tk(N) 
        bm = bm[:, :, :(self.k+1)]
        if self.tk == self.pi[-2]:
            Y_tk_1 = torch.sum(self.chaos_coeffs.unsqueeze(0) * X_tk_1, dim=1).unsqueeze(1).unsqueeze(1)
        else:
            with torch.no_grad():
                Y_tk_1 = self.Y_NN_tk_1(X_tk_1) 
                
        if self.k > 0:
            Z_tk = self.Z_NN_tk(X_tk)      
            Y_tk = self.Y_NN_tk(X_tk)
        else:
            Z_tk = self.Z0.unsqueeze(0).repeat(N, 1, 1)
            Y_tk = None
        # We are using the explicit scheme
        
        f_tk = self.f_BSDE(self.tk, Y_tk_1, Z_tk, bm)
        return Z_tk, Y_tk, Y_tk_1, f_tk, DeltaBm_tk

    def train_ZNN_tk(self, batch_size = int(1e4), batch_size_test = int(1e4), n_steps = int(1e3), n_eval = 10, patience = 5):
        
        trange = tqdm(range(n_steps), bar_format="{l_bar}{bar:10}{r_bar}")

        if self.k > 0:
            patience_it = 0
            Z_tk, Y_tk_1, DeltaBm_tk = self.sample_for_training_Z_tk(batch_size)
            loss_Z = loss_function_Z(Z_tk, Y_tk_1, DeltaBm_tk, self.Delta_tk)
            best_loss = np.inf
            for step in trange:
            
                if ((step+1) % (n_steps // n_eval) == 0) or step == 0:
                    with torch.no_grad():
                        Z_tk, Y_tk_1, DeltaBm_tk = self.sample_for_training_Z_tk(batch_size_test)
                    loss_Z_test = loss_function_Z(Z_tk, Y_tk_1, DeltaBm_tk, self.Delta_tk)
                    if loss_Z_test < best_loss: 
                        best_params = self.Z_NN_tk.state_dict()
                        best_loss = loss_Z_test
                        patience_it = 0
                    else:
                        patience_it += 1 
                    
                    loss_Z = format(loss_Z.item(), ".4e")
                    loss_Z_test = format(loss_Z_test.item(), ".4e")

                    tqdm.write(f'Step [{step}/{n_steps}], Training Loss: {loss_Z}, Validation Loss: {loss_Z_test}')    

                    if patience_it == patience: 
                        break
    
                self.opt_Z_tk.zero_grad()
                Z_tk, Y_tk_1, DeltaBm_tk = self.sample_for_training_Z_tk(batch_size)
                loss_Z = loss_function_Z(Z_tk, Y_tk_1, DeltaBm_tk, self.Delta_tk)
                loss_Z.backward()
                self.opt_Z_tk.step()
    
            self.Z_NN_tk.load_state_dict(best_params)

        else:
            Z0 = torch.zeros((n_steps, self.d, 1), dtype=torch.float32, device=self.device)
            for i in trange:
                _, Y_tk_1, DeltaBm_tk = self.sample_for_training_Z_tk(batch_size)
                Z0[i] = torch.mean(Y_tk_1 * (DeltaBm_tk/self.Delta_tk), dim=0)
            self.Z0 = torch.mean(Z0, dim=0)

    def train_YNN_tk(self, batch_size = int(1e4), batch_size_test = int(1e4), n_steps = int(1e3), n_eval = 10, patience = 5):
        
        trange = tqdm(range(n_steps), bar_format="{l_bar}{bar:10}{r_bar}")

        if self.k > 0:
            patience_it = 0
            Z_tk, Y_tk, Y_tk_1, f_tk, DeltaBm_tk = self.sample_for_training_Y_tk(batch_size)
            loss_Y = loss_function_Y(Y_tk, Z_tk, Y_tk_1, DeltaBm_tk, f_tk, self.Delta_tk)
            best_loss = np.inf
            for step in trange:
                if ((step+1) % (n_steps // n_eval) == 0) or step == 0:
                    with torch.no_grad():
                        Z_tk, Y_tk, Y_tk_1, f_tk, DeltaBm_tk = self.sample_for_training_Y_tk(batch_size_test)
                    loss_Y_test = loss_function_Y(Y_tk, Z_tk, Y_tk_1, DeltaBm_tk, f_tk, self.Delta_tk)
                    if loss_Y_test < best_loss: 
                        best_params = self.Y_NN_tk.state_dict()
                        best_loss = loss_Y_test
                        patience_it = 0
                    else:
                        patience_it += 1 
                    
                    loss_Y = format(loss_Y.item(), ".4e")
                    loss_Y_test = format(loss_Y_test.item(), ".4e")

                    tqdm.write(f'Step [{step}/{n_steps}], Training Loss: {loss_Y}, Validation Loss: {loss_Y_test}')    
                
                    if patience_it == patience: 
                        break
    
                self.opt_Y_tk.zero_grad()
                Z_tk, Y_tk, Y_tk_1, f_tk, DeltaBm_tk = self.sample_for_training_Y_tk(batch_size)
                loss_Y = loss_function_Y(Y_tk, Z_tk, Y_tk_1, DeltaBm_tk, f_tk, self.Delta_tk)
                loss_Y.backward()
                self.opt_Y_tk.step()
    
            self.Y_NN_tk.load_state_dict(best_params)

        else:
            Y0 = torch.zeros((n_steps, 1, 1), dtype=torch.float32, device=self.device)
            for i in trange:
                Z_tk, Y_tk, Y_tk_1, f_tk, DeltaBm_tk = self.sample_for_training_Y_tk(batch_size)
                Y0[i] = torch.mean(Y_tk_1 + self.Delta_tk * f_tk, dim=0)
            self.Y0 = torch.mean(Y0, dim=0).to(self.device)

    def check_error_Euler_tk(self, N):
        """
        This function checks the error in the approximation of Y and Z by computing 
        E|Y_tk - Y_tk+1 - hf(tk, Y_tk+1, Z_tk) + Z_tk (B_tk+1 - B_tk)|^2
        """
        with torch.no_grad():
            Z_tk, Y_tk, Y_tk_1, f_tk, DeltaBm_tk = self.sample_for_training_Y_tk(N)
        if self.k>0:
            error = torch.mean(torch.abs((Y_tk - Y_tk_1 - self.Delta_tk*f_tk + torch.matmul(Z_tk.unsqueeze(1), DeltaBm_tk.unsqueeze(2)).squeeze(-1)))**2)
        else:
            Y_tk = torch.ones((N, 1, 1), device=self.device)*self.Y0
            error = torch.mean(torch.abs((Y_tk - Y_tk_1 - self.Delta_tk*f_tk + torch.matmul(Z_tk.unsqueeze(1), DeltaBm_tk.unsqueeze(2)).squeeze(-1)))**2)

        error = format(error.item(), ".4e")
        print(f'Mean Euler Error: {error}')
        return error
