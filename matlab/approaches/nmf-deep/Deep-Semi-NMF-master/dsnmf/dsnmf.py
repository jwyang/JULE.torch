from __future__ import print_function
from collections import OrderedDict

import numpy as np
import theano
import theano.tensor as T

from scipy.sparse.linalg import svds

relu = lambda x: 0.5 * (x + abs(x))

def floatX(x):
    return np.asarray(x, dtype=theano.config.floatX)

def appr_seminmf(M, r):
    """
        Approximate Semi-NMF factorisation. 
        
        Parameters
        ----------
        M: array-like, shape=(n_features, n_samples)
        r: number of components to keep during factorisation
    """
    
    if r < 2:
        raise ValueError("The number of components (r) has to be >=2.")

    A, S, B = svds(M, r-1)
    S = np.diag(S)
    A = np.dot(A, S)
 
    m, n = M.shape
 
    for i in range(r-1):
        if B[i, :].min() < (-B[i, :]).min():
            B[i, :] = -B[i, :]
            A[:, i] = -A[:, i]
            
            
    if r == 2:
        U = np.concatenate([A, -A], axis=1)
    else:
        An = -np.sum(A, 1).reshape(A.shape[0], 1)
        U = np.concatenate([A, An], 1)
    
    V = np.concatenate([B, np.zeros((1, n))], 0)

    if r>=3:
        V -= np.minimum(0, B.min(0))
    else:
        V -= np.minimum(0, B)

    return U, V
    
def adam(loss, params, learning_rate=0.001, beta1=0.9,
         beta2=0.999, epsilon=1e-8):
    """Adam updates

    Adam updates implemented as in [1]_.

    Parameters
    ----------
    loss_or_grads : symbolic expression or list of expressions
        A scalar loss expression, or a list of gradient expressions
    params : list of shared variables
        The variables to generate update expressions for
    learning_rate : float
        Learning rate
    beta_1 : float
        Exponential decay rate for the first moment estimates.
    beta_2 : float
        Exponential decay rate for the second moment estimates.
    epsilon : float
        Constant for numerical stability.

    Returns
    -------
    OrderedDict
        A dictionary mapping each parameter to its update expression

    Notes
    -----
    The paper [1]_ includes an additional hyperparameter lambda. This is only
    needed to prove convergence of the algorithm and has no practical use
    (personal communication with the authors), it is therefore omitted here.

    References
    ----------
    .. [1] Kingma, Diederik, and Jimmy Ba (2014):
           Adam: A Method for Stochastic Optimization.
           arXiv preprint arXiv:1412.6980.
    """
         
    all_grads = theano.grad(loss, params)
    t_prev = theano.shared(floatX(0.))
    updates = OrderedDict()

    for param, g_t in zip(params, all_grads):
        m_prev = theano.shared(param.get_value() * 0.)
        v_prev = theano.shared(param.get_value() * 0.)
        t = t_prev + 1
        m_t = beta1*m_prev + (1-beta1)*g_t
        v_t = beta2*v_prev + (1-beta2)*g_t**2
        a_t = learning_rate*T.sqrt(1-beta2**t)/(1-beta1**t)
        step = a_t*m_t/(T.sqrt(v_t) + epsilon)

        updates[m_prev] = m_t
        updates[v_prev] = v_t
        updates[param] = param - step

    updates[t_prev] = t
    return updates


def init_weights(X, num_components, svd_init=True):
    if svd_init:
        return appr_seminmf(X, num_components)

    Z = 0.08 * np.random.rand(X.shape[0], num_components)
    H = 0.08 * np.random.rand(num_components, X.shape[1])

    return Z, H


from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
rng = RandomStreams()

def dropout(x, p=0):
    if p == 0:
        return x
    else:
        p = 1 - p
        x /= p

        return x * rng.binomial(x.shape, p=p, dtype=theano.config.floatX)
        

class DSNMF(object):

    def __init__(self, data, layers, verbose=False, l1_norms=[], pretrain=True, learning_rate=1e-3):
        """
        Parameters
        ----------
        :param data: array-like, shape=(n_samples, n_features)
        :param layers: list, shape=(n_layers) containing the size of each of the layers
        :param verbose: boolean
        :param l1_norms: list, shape=(n_layers) the l1-weighting of each of the layers
        :param pretrain: pretrain layers using svd
        """
        H = data.T
        
        assert len(layers) > 0, "You have to provide a positive number of layers."

        params = []

        for i, l in enumerate(layers, start=1):
            print('Pretraining {}th layer [{}]'.format(i, l), end='\r')

            Z, H = init_weights(H, l, svd_init=pretrain)

            params.append(theano.shared(floatX(Z), name='Z_%d' % (i)))

        params.append(theano.shared(floatX(H), name='H_%d' % len(layers)))

        self.params = params
        self.layers = layers
        
        cost = ((data.T - self.get_h(-1))**2).sum()
        
        for norm, param in zip(l1_norms, params):
            cost += ((abs(param)) * norm).sum()
    
        H = relu(self.params[-1])
        
        updates = adam(cost, params, learning_rate=learning_rate)

        self.cost = cost
        self.train_fun = theano.function([], cost, updates=updates)
        self.get_features = theano.function([], H)

        self.get_reconstruction = theano.function([], self.get_h(-1))

    def finetune_features(self):

        updates = adam(self.cost, self.params[-1:])
        self.train_fun = theano.function([], self.cost, updates=updates)

    def get_param_values(self):
        return [p.get_value() for p in self.params]

    def set_param_values(self, values):
        params = self.params

        if len(params) != len(values):
            raise ValueError("mismatch: got %d values to set %d parameters" %
                            (len(values), len(params)))

        for p, v in zip(params, values):
            if p.get_value().shape[0] != v.shape[0]:
                raise ValueError("mismatch: parameter has shape %r but value to "
                             "set has shape %r" %
                             (p.get_value().shape, v.shape))
            else:
                p.set_value(v)

    def get_h(self, layer_num, have_dropout=False):
        h = relu(self.params[-1])
        
        if have_dropout:
            h = dropout(h, p=.1)
            
        for z in reversed(self.params[1:-1][:]):
            h = relu(z.dot(h))

        if layer_num == -1:
            h = self.params[0].dot(h)
        
        return h
        
