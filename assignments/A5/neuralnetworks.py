import numpy as np
import mlutilities as ml
import optimizers as opt
import matplotlib.pyplot as plt
import copy
import time


class NeuralNetwork:

    def __init__(self, n_inputs, n_hiddens_list, n_outputs):

        if not isinstance(n_hiddens_list, list):
            raise Exception('NeuralNetwork: n_hiddens_list must be a list or tuple')
 
        if len(n_hiddens_list) == 0:
            self.n_hidden_layers = 0
        elif n_hiddens_list[0] == 0:
            self.n_hidden_layers = 0
        else:
            self.n_hidden_layers = len(n_hiddens_list)
            
        self.n_inputs = n_inputs
        self.n_hiddens_list = n_hiddens_list
        self.n_outputs = n_outputs
        
        # Do we have any hidden layers?
        self.Vs = []
        ni = n_inputs
        for layeri in range(self.n_hidden_layers):
            n_in_layer = self.n_hiddens_list[layeri]
            self.Vs.append(1 / np.sqrt(1 + ni) * np.random.uniform(-1, 1, size=(1 + ni, n_in_layer)))
            ni = n_in_layer
        self.W = 1/np.sqrt(1 + ni) * np.random.uniform(-1, 1, size=(1 + ni, n_outputs))

        # Member variables for standardization
        self.Xmeans = None
        self.Xstds = None
        self.Tmeans = None
        self.Tstds = None
        
        self.trained = False
        self.reason = None
        self.error_trace = None
        self.n_epochs = None
        self.training_time = None

    def __repr__(self):
        str = f'NeuralNetwork({self.n_inputs}, {self.n_hiddens_list}, {self.n_outputs})'
        if self.trained:
            str += f'\n   Network was trained for {self.n_epochs} epochs'
            str += f' that took {self.training_time:.4f} seconds. Final error is {self.error_trace[-1]}'
        else:
            str += '  Network is not trained.'
        return str

    def _standardizeX(self, X):
        result = (X - self.Xmeans) / self.XstdsFixed
        result[:, self.Xconstant] = 0.0
        return result

    def _unstandardizeX(self, Xs):
        return self.Xstds * Xs + self.Xmeans

    def _standardizeT(self, T):
        result = (T - self.Tmeans) / self.TstdsFixed
        result[:, self.Tconstant] = 0.0
        return result

    def _unstandardizeT(self, Ts):
        return self.Tstds * Ts + self.Tmeans

    def _pack(self, Vs, W):
        return np.hstack([V.flat for V in Vs] + [W.flat])

    def _unpack(self, w):
        first = 0
        n_this_layer = self.n_inputs
        for i in range(self.n_hidden_layers):
            self.Vs[i][:] = w[first:first + (1 + n_this_layer) * 
                              self.n_hiddens_list[i]].reshape((1 + n_this_layer, self.n_hiddens_list[i]))
            first += (1 + n_this_layer) * self.n_hiddens_list[i]
            n_this_layer = self.n_hiddens_list[i]
        self.W[:] = w[first:].reshape((1 + n_this_layer, self.n_outputs))

    def _objectiveF(self, w, X, T):
        self._unpack(w)
        # Do forward pass through all layers
        Zprev = X
        for i in range(self.n_hidden_layers):
            V = self.Vs[i]
            Zprev = np.tanh(Zprev @ V[1:, :] + V[0:1, :])  # handling bias weight without adding column of 1's
        Y = Zprev @ self.W[1:, :] + self.W[0:1, :]
        return 0.5 * np.mean((T - Y)**2)

    def _gradientF(self, w, X, T):
        self._unpack(w)
        # Do forward pass through all layers
        Z_prev = X  # output of previous layer
        Z = [Z_prev]
        for i in range(self.n_hidden_layers):
            V = self.Vs[i]
            Z_prev = np.tanh(Z_prev @ V[1:, :] + V[0:1, :])
            Z.append(Z_prev)
        Y = Z_prev @ self.W[1:, :] + self.W[0:1, :]
        # Do backward pass, starting with delta in output layer
        delta = -(T - Y) / (X.shape[0] * T.shape[1])
        # Another way to define dEdW without calling np.insert                                                                         
        dW = np.vstack((np.sum(delta, axis=0), Z[-1].T @ delta))
        dVs = []
        delta = (1 - Z[-1]**2) * (delta @ self.W[1:, :].T)
        for Zi in range(self.n_hidden_layers, 0, -1):
            Vi = Zi - 1  # because X is first element of Z
            dV = np.vstack((np.sum(delta, axis=0), Z[Zi-1].T @ delta))
            dVs.insert(0, dV)  # like append, but at front of list of dVs
            delta = (delta @ self.Vs[Vi][1:, :].T) * (1 - Z[Zi-1]**2)
        return self._pack(dVs, dW)

    def train(self, X, T, n_epochs, verbose=False, save_weights_history=False):
        
        if self.Xmeans is None:
            self.Xmeans = X.mean(axis=0)
            self.Xstds = X.std(axis=0)
            self.Xconstant = self.Xstds == 0
            self.XstdsFixed = copy.copy(self.Xstds)
            self.XstdsFixed[self.Xconstant] = 1
        X = self._standardizeX(X)

        if T.ndim == 1:
            T = T.reshape((-1, 1))

        if self.Tmeans is None:
            self.Tmeans = T.mean(axis=0)
            self.Tstds = T.std(axis=0)
            self.Tconstant = self.Tstds == 0
            self.TstdsFixed = copy.copy(self.Tstds)
            self.TstdsFixed[self.Tconstant] = 1
        T = self._standardizeT(T)

        start_time = time.time()

        scgresult = opt.scg(self._pack(self.Vs, self.W),
                            self._objectiveF, self._gradientF,
                            [X, T],
                            n_iterations=n_epochs,
                            verbose=verbose,
                            save_wtrace=save_weights_history)

        self._unpack(scgresult['w'])
        self.reason = scgresult['reason']
        self.error_trace = np.sqrt(scgresult['ftrace']) # * self.Tstds # to _unstandardize the MSEs
        self.n_epochs = len(self.error_trace)
        self.trained = True
        self.weight_history = scgresult['wtrace'] if save_weights_history else None
        self.training_time = time.time() - start_time
        return self

    def use(self, X, all_outputs=False):
        Zprev = self._standardizeX(X)
        Z = [Zprev]
        for i in range(self.n_hidden_layers):
            V = self.Vs[i]
            Zprev = np.tanh(Zprev @ V[1:, :] + V[0:1, :])
            Z.append(Zprev)
        Y = Zprev @ self.W[1:, :] + self.W[0:1, :]
        Y = self._unstandardizeT(Y)
        return (Y, Z[1:]) if all_outputs else Y

    def get_n_epochs(self):
        return self.n_epochs

    def get_error_trace(self):
        return self.error_trace

    def get_training_time(self):
        return self.training_time

    def get_weight_history(self):
        return self.weight_history

    def draw(self, input_names=None, output_names=None, gray=False):
        ml.draw(self.Vs, self.W, input_names, output_names, gray)
 
if __name__ == '__main__':

    X = np.arange(10).reshape((-1, 1))
    T = X + 2

    nnet = NeuralNetwork(1, [], 1)
    nnet.train(X, T, 10)
    print(nnet)
    Y = nnet.use(X)
    print(np.hstack((T, Y)))
    
    nnet = NeuralNetwork(1, [0], 1)
    nnet.train(X, T, 10)
    print(nnet)
    Y = nnet.use(X)
    print(np.hstack((T, Y)))
    
    nnet = NeuralNetwork(1, [5, 5], 1)
    nnet.train(X, T, 100)
    print(nnet)
    Y = nnet.use(X)
    print(np.hstack((T, Y)))

    nnet.draw()
