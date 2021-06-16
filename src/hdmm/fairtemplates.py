from . import matrix, workload, error, mechanism, templates
from functools import reduce
import numpy as np
from scipy import optimize
from scipy import sparse
from scipy.sparse.linalg import spsolve_triangular
from scipy.linalg.lapack import dpotrf, dpotri

class TemplateStrategy:

    def __init__(self, seed=None):
        if seed is None:
            seed = np.random.randint(2**32-1)

        self.prng = np.random.RandomState(seed)
        
    def strategy(self):
        pass
  
    def _AtA1(self):
        return self.strategy().gram().pinv().dense_matrix()
 
    def _loss_and_grad(self, params):
        pass

    def _set_workload(self, W, indAs=None):
        self._workload = W
        self._gram = W.gram()

    def optimize(self, W, init=None, indAs=None):
        """
        Optimize strategy for given workload 
        :param W: the workload, may be a n x n numpy array for WtW or a workload object
        """
        self._set_workload(W, indAs=indAs)
        if init is None:
            init = self.prng.rand(self._params.size)
        bnds = [(0,None)]*init.size
       
        opts = { 'ftol' : 1e-4 }
        res = optimize.minimize(self._loss_and_grad, init, jac=True, method='L-BFGS-B', bounds=bnds, options=opts)
        self._params = np.maximum(0, res.x)
        loss, grad = self._loss_and_grad(self._params)
        return res.fun, loss



    def restart_optimize(self, W, restarts):
        best_A, best_params, best_loss = None, None, np.inf
        init = self._params
        for _ in range(restarts):
            loss = self.optimize(W, init)
            A = self.strategy()
            #loss = error.rootmse(W, A)
            if loss <= best_loss:
                best_loss = loss
                best_A = A
                best_params = np.copy(self._params)
            init = np.random.rand(self._params.size)
        self._params = best_params
        return best_A, best_loss

    def priv_restart_optimize(self, W, restarts):
        best_A, best_params, best_loss = None, None, np.inf
        init = self._params
        for _ in range(restarts):
            loss = self.priv_optimize(W, init)
            A = self.strategy()
            #loss = error.rootmse(W, A)
            if loss <= best_loss:
                best_loss = loss
                best_A = A
                best_params = np.copy(self._params)
            init = np.random.rand(self._params.size)
        self._params = best_params
        return best_A, best_loss



class PIdentity(TemplateStrategy):
    """
    A PIdentity strategy is a strategy of the form (I + B) D where D is a diagonal scaling matrix
    that depends on B and ensures uniform column norm.  B is a p x n matrix of free parameters.
    """
    def __init__(self, p, n, seed=None, mode='max'):
        """
        Initialize a PIdentity strategy
        :param p: the number of non-identity queries
        :param n: the domain size
        """
        super(PIdentity, self).__init__(seed)

        self._params = self.prng.rand(p*n)
        self.p = p
        self.n = n
        self.mode = mode

    def strategy(self):
        B = sparse.csr_matrix(self._params.reshape(self.p, self.n))
        I = sparse.eye(self.n, format='csr')
        A = sparse.vstack([I, B], format='csr')
        return matrix.EkteloMatrix(A / A.sum(axis=0))

    def _AtA1(self):
        B = np.reshape(self._params, (self.p,self.n))
        scale = 1.0 + np.sum(B, axis=0)
        R = np.linalg.inv(np.eye(self.p) + B @ B.T) # O(k^3)
        return (np.eye(self.n) - B.T @ R @ B)*scale*scale[:,None]
 
    def _set_workload(self, W, indAs=None):
        #self._WtW = W.gram().dense_matrix()
        temp = []
        #weights = []
        inderrs = []
        #maximum = 0
        n = self.n

        for ind, i in enumerate(W):
            temp.append(i.gram().dense_matrix())
            #weights.append(1/np.linalg.norm(i.dense_matrix()))
            #if maximum < np.linalg.norm(i.dense_matrix()):
             #   maximum = np.linalg.norm(i.dense_matrix())
            inderr = error.expected_error(i, indAs[ind], eps=1)
            inderr *= len(W)**2/2
            inderrs.append(inderr)
            #weights.append(1/(np.max(np.abs(i).sum(axis=0)) ) )

        #weights = [i * maximum for i in weights]
        self._WTW= temp
        #self.weights =weights
        self.inderrs = inderrs
        
        
        
    def _loss_and_grad(self, params):
        WtW = self._WTW
        p, n = self.p, self.n
        k = len(WtW)
        grads = []
        losses = []
        inderrs = self.inderrs
        
        B = np.reshape(params, (p,n))
        #initial values of the p weight rows
        scale = 1.0 + np.sum(B, axis=0)
        # scale of every row (query)
        
        for i in WtW:

            #swap these to out to swicth between changing the frob norm and column norm
            #weights.append(1/(np.max(np.abs(i).sum(axis=0)) ) )
            #weights.append(1/(np.linalg.norm(i)))
            #weights.append(1)
            

            try: R = np.linalg.inv(np.eye(p) + B.dot(B.T)) # O(k^3)
            except: return np.inf, np.zeros_like(params)
            C = i * scale * scale[:,None] # O(n^2)

            M1 = R.dot(B) # O(n k^2)
            M2 = M1.dot(C) # O(n^2 k)
            M3 = B.T.dot(M2) # O(n^2 k)
            M4 = B.T.dot(M2.dot(M1.T)).dot(B) # O(n^2 k)

            Z = -(C - M3 - M3.T + M4) * scale * scale[:,None] # O(n^2)

            Y1 = 2*np.diag(Z) / scale # O(n)
            Y2 = 2*(B/scale).dot(Z) # O(n^2 k)
            g = Y1 + (B*Y2).sum(axis=0) # O(n k)

            loss = np.trace(C) - np.trace(M3)
            grad = (Y2*scale - g) / scale**2
                
            grads.append(grad)
            losses.append(loss)
        




            #investigation print statements
            #print(np.trace(M3))
            #print(M3)
            #print(np.shape(M3))

        losses = np.asarray(losses)
        inderrs = np.asarray(inderrs)
        

        if self.mode == 'max':
            losses = np.divide(losses, inderrs)
            index = np.argmax(losses)
            return losses[index], grads[index].flatten()
        elif self.mode == 'sum':
            losses = np.divide(losses, inderrs)
            losses = np.sum(losses)
            grads = np.asarray(grads)
            for i in range(inderrs.shape[0]):
                grads[i] /= inderrs[i]
            grads = np.sum(grads, axis = 0)
            return losses, grads.flatten()
        elif self.mode == 'diff':
            losses = np.subtract(losses, inderrs)
            index = np.argmax(losses)
            return losses[index], grads[index].flatten()
