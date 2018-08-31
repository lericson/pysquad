import time
import logging
from contextlib import contextmanager

import numpy as np
import numba as nb

from squad.traj import Trajectory


log = logging.getLogger(__name__)


class Learner():
    def __init__(self, *, model, T, dt):
        self.model = model
        self.T = T
        self.dt = dt

    def fit(self, states, actions):
        raise NotImplementedError

    def predict(self, states):
        raise NotImplementedError

    def predict_one(self, state):
        return self.predict([state])[0]

    def validate(self, states, actions):
        actions_pred = self.predict(states)
        return np.linalg.norm(actions - actions_pred, axis=1)

    def unroll(self, states):
        N = len(states)
        X = np.empty((N, self.T+1, *self.model.state_shape))
        U = np.empty((N, self.T, *self.model.action_shape))
        R = np.empty((N, self.T,))

        X[:, 0] = states
        for i in range(self.T):
            U[:, i] = self.predict(X[:, i])
            U[:, i] = np.clip(U[:, i], self.model.u_lower, self.model.u_upper)
            for k, (xki, uki) in enumerate(zip(X[:, i], U[:, i])):
                _, X[k, i+1], R[k, i] = self.model.step(xki, uki, dt=self.dt)

        log.info('state mean:\n%s', self.model.state_rep(X.mean(axis=(0, 1))))
        log.info('state variance:\n%s', self.model.state_rep(X.var(axis=(0, 1))))
        U_stack = U.reshape((-1, *self.model.action_shape))
        U_centered = U_stack - U_stack.mean(axis=1, keepdims=True)
        U_cov = U_centered.T @ U_centered
        log.info('action mean: %s', U.mean(axis=(0, 1)))
        log.info('action covariance:\n%s', U_cov)

        return [Trajectory(dt=self.dt, states=Xk, actions=Uk, rewards=Rk)
                for Xk, Uk, Rk in zip(X, U, R)]

    @contextmanager
    def transaction(self):
        yield


class LeastSquaresLearner(Learner):
    "Least squares regression"

    def __init__(self, *, theta=None, **kwds):
        super().__init__(**kwds)
        self.theta = theta

    def fit(self, states, actions):
        actions = np.asarray(actions)
        states_proj = self.project(states)
        self.theta, eps, *_ = np.linalg.lstsq(states_proj, actions, rcond=None)
        log.info('fit done, eps: %s', eps)
        return eps

    def predict(self, states):
        u = self.project(states) @ self.theta
        return np.clip(u, self.model.u_lower, self.model.u_upper)


class PolynomialLeastSquares(LeastSquaresLearner):
    "Regression on a_n x^n + a_{n-1} x^{n-1} + ... + a_0"

    order = 2
    proj_matrix = None

    def project(self, states):
        states_proj = np.atleast_2d(states)
        if self.proj_matrix is not None:
            states_proj = states @ self.proj_matrix
        N, K = states_proj.shape
        ones = np.ones((N, 1))
        states_proj = np.hstack((ones, states_proj))
        K += 1
        assert self.order == 2, 'only support order = 2'
        out = np.empty((N, K*(K+1)//2))
        start = 0
        for i in range(K):
            stop = start + K - i
            out[:, start:stop]  = np.repeat(states_proj[:, i:i+1], K-i, axis=1)
            out[:, start:stop] *= states_proj[:, i:]
            start = stop
        return out
        #return np.hstack((ones, *(states_proj**i for i in range(1, self.order+1))))


class GradientDescentLearner(Learner):
    num_epochs = 10
    batch_size = 64
    stochastic = True
    proj_matrix = None

    def __init__(self, *, theta=None, order=2, **kwds):
        super().__init__(**kwds)
        self.theta = theta
        self.order = order

    def project(self, states):
        states_proj = np.atleast_2d(states)
        if self.proj_matrix is not None:
            states_proj = states @ self.proj_matrix
        ones = np.ones((states_proj.shape[0], 1))
        return np.hstack((ones, *(states_proj**i for i in range(1, self.order+1))))

    def predict(self, states):
        if np.any(np.isnan(states)):
            raise ValueError(states)
        return self.forward(self.theta, self.project(states))

    def fit(self, states, actions):
        states, actions = np.asarray(states), np.asarray(actions)
        for i in range(self.num_epochs):
            self.fit_epoch(states, actions, epoch=i)

    def fit_epoch(self, states, actions, epoch=None):
        loss, t0 = 0.0, time.time()
        for mb_states, mb_actions in self.batches(states, actions):
            mb_loss = self._fit_minibatch(self.project(mb_states), mb_actions)
            loss += mb_loss
        t1 = time.time()
        loss /= len(states)
        log.debug('loss: %.3g (%.3g inst/s)', loss, states.shape[0]/(t1 - t0))
        return loss

    def batches(self, states, actions):
        n = states.shape[0]
        assert actions.shape[0] == n
        idxs = np.arange(n)
        if self.stochastic:
            np.random.shuffle(idxs)
        out_states = np.empty((self.batch_size, states.shape[1]))
        out_actions = np.empty((self.batch_size, actions.shape[1]))
        for start in range(0, n, self.batch_size):
            stop = start + self.batch_size
            batch_idxs = idxs[start:stop]
            if np.size(batch_idxs, 0) == 0:
                log.warn('empty batch')
                break
            out_states[:batch_idxs.shape[0]]  = states[batch_idxs]
            out_actions[:batch_idxs.shape[0]] = actions[batch_idxs]
            yield out_states, out_actions


class MultiLayerPerceptron(GradientDescentLearner):
    def __init__(self, *, num_units, **kwds):
        super().__init__(**kwds)
        sizes = list(zip(num_units[1:], num_units[:-1]))
        if self.theta is not None:
            assert [wl.shape for wl in self.theta] == sizes
        else:
            self.theta = [np.random.normal(0.0, 1e-2, layer_size)
                          for layer_size in sizes]
            self.theta = tuple(self.theta)

    def _fit_minibatch(self, states, actions, *, learning_rate=0.001, gamma=0.9, epsilon=1e-8):
        df, loss = self.dlossdtheta(self.theta, states, actions)

        # RMSprop
        if not hasattr(self, 'g2'):
            self.g2 = [np.zeros_like(dfidw) for dfidw in df]
        self.g2 = [gamma*gi2 + (1.0 - gamma)*dfidw**2
                   for gi2, dfidw in zip(self.g2, df)]
        df = [dfidw/np.sqrt(gi2 + epsilon)
              for gi2, dfidw in zip(self.g2, df)]

        # clipnorm
        #df = [dfidw/max(1.0, np.linalg.norm(dfidw)) for dfidw in df]

        self.theta = tuple(layer_w - learning_rate * dfidw
                           for layer_w, dfidw in zip(self.theta, df))

        return loss

    def scope_explicit():
        # output = scale*(translate + tanh(z))
        #scale, translate, shift = 0.5, 1.0, 1.0
        #scale, translate = 1.0, 0.0

        @nb.njit(cache=True)
        def forward(theta, states):
            z0 = np.dot(theta[0], states.T)
            a0 = z0
            zero_inds = z0 < 0.0
            for i in range(a0.shape[1]):
                a0[zero_inds[:, i], i] *= 1e-2
            z1 = np.dot(theta[1], a0)
            #a1 = scale*(translate + np.tanh(z1 - shift))
            a1 = z1
            return a1

        @nb.njit(cache=True)
        def loss(theta, states, actions):
            y_pred = forward(theta, states)
            err = y_pred - actions.T
            sq_err = 0.5*err**2
            return np.mean(np.sum(sq_err, axis=(0,)), axis=(0,))

        @nb.njit(cache=True)
        def dlossdtheta(theta, states, actions):
            n = actions.shape[0]
            z0 = np.dot(theta[0], states.T)
            a0 = z0.copy()
            zero_inds = z0 < 0.0
            for i in range(n):
                a0[zero_inds[:, i], i] *= 1e-2

            z1 = np.dot(theta[1], a0)
            #a1 = scale*(translate + np.tanh(z1 - shift))
            a1 = z1

            err = a1 - actions.T
            dlossda1 = err
            #da1dz1 = scale*(1 - np.tanh(z1 - shift)**2)
            da1dz1 = np.ones_like(z1)
            delta1 = dlossda1 * da1dz1
            delta0 = theta[1].T @ delta1
            for i in range(n):
                delta0[zero_inds[:, i], i] *= 1e-2

            dlossdtheta = (delta0 @ states / n,
                           delta1 @ a0.T / n)

            sq_err = err**2
            loss = 0.5*np.sum(np.sum(sq_err, axis=0), axis=0)/n

            return (dlossdtheta, loss)

        return loss, forward, dlossdtheta

    def scope_tangent():
        def forward(theta, states):
            states_T = np.transpose(states)
            z0 = np.dot(theta[0], states_T)
            a0 = z0
            zero_inds = z0 < 0.0
            a0[zero_inds] *= 1e-2
            z1 = np.dot(theta[1], a0)
            #a1 = 0.5*(1.0 + np.tanh(z1))
            a1 = z1
            return a1

        def loss(theta, states, actions):
            y_pred = forward(theta, states)
            actions_T = np.transpose(actions)
            err = y_pred - actions_T
            sq_err = err**2
            return 0.5*np.mean(np.sum(sq_err, axis=(0,)), axis=(0,))

        import tangent
        dlossdtheta = tangent.grad(loss, preserve_result=True)

        return loss, forward, dlossdtheta

    loss, forward, dlossdtheta = map(staticmethod, scope_explicit())


class TwoLayerPerceptron(GradientDescentLearner):

    proj_matrix = None
    num_epochs = 20
    batch_size = 64
    order = 1

    def __init__(self, *, state_shape, num_epochs=10, num_hidden=(150, 150, 150),
                 learning_rate=1e-4, **kwds):
        super().__init__(**kwds)
        import torch
        modules = []
        dims = [*state_shape, *num_hidden, *self.model.action_shape]
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            modules.append(torch.nn.Linear(in_dim, out_dim))
            modules.append(torch.nn.ELU())
        modules[-1] = torch.nn.Tanh()
        self.nn = torch.nn.Sequential(*modules).float()
        self.lossfn = torch.nn.MSELoss()
        self.opt = torch.optim.Adam(self.nn.parameters(), lr=learning_rate)

    def project(self, states):
        states_proj = np.atleast_2d(states)
        if self.proj_matrix is not None:
            states_proj = states @ self.proj_matrix
        return states_proj

    def predict(self, states):
        import torch
        if np.any(np.isnan(states)):
            raise ValueError(states)
        states_tensor = torch.as_tensor(self.project(states),
                                        dtype=torch.float)
        return self.nn(states_tensor).detach().numpy()

    def _fit_minibatch(self, states, actions):
        import torch
        states_tensor  = torch.as_tensor(states, dtype=torch.float)
        actions_tensor = torch.as_tensor(actions, dtype=torch.float)
        actions_pred   = self.nn(states_tensor)
        loss = self.lossfn(actions_pred, actions_tensor)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return loss.item()

    def validate(self, states, actions):
        import torch
        states_tensor  = torch.as_tensor(self.project(states), dtype=torch.float)
        actions_tensor = torch.as_tensor(actions, dtype=torch.float)
        actions_pred   = self.nn(states_tensor)
        return self.lossfn(actions_pred, actions_tensor).item()

    @contextmanager
    def transaction(self):
        import torch
        state_dict = {k: torch.tensor(v) for k, v in self.nn.state_dict().items()}
        def rollback():
            self.nn.load_state_dict(state_dict)
        yield rollback
