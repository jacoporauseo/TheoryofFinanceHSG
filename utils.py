import numpy as np


class MVInterface:
    def maxExpUtility(self):
        pass
    def capitalMarketLine(self):
        pass
    def minimumVariancePortfolio(self):
        pass
    def computeExpectedUtility(self):
        pass


class MeanVariance(MVInterface):
    def __init__(self, mu, R_f, Sigma):
        self.mu = mu
        self.R_f = R_f
        self.Sigma = Sigma
        self.k = 0
        self.mu_e = self.mu - np.ones(len(self.mu))*self.R_f
        self.S_inv = np.linalg.inv(self.Sigma) 

    def setk(self,k):
        """Set a Risk Aversion rate"""
        self.k = k

    def computeExpectedUtility(self,w):
        """For a given set of weights compute the expected utility"""
        return (w*self.mu + (1-w)*self.R_f) - self.k/2*(w**2)*self.Sigma
    
    def maxExpUtility(self):
        """Compute the optimal weights"""
        w_star = 1/self.k * np.linalg.inv(self.Sigma) @ self.mu_e
        w_rf = 1 - w_star @ np.ones(len(w_star))
        return w_star, w_rf
    
    def capitalMarketLine(self, variance_array = None):
        """lambdas: array of scalars controlling exposure to the tangent portfolio"""
        w_t = self.tangentPortfolio()
        mu_t, var_t = self.meanAndVariance(w_t)

        if variance_array == None:
            variance_array = np.linspace(0,1.5*var_t, 100)

        mu = self.R_f + (mu_t - self.R_f)/np.sqrt(var_t) * np.sqrt(variance_array)

        return np.sqrt(variance_array), mu
    
    def tangentPortfolio(self):
        """Returns the Tangent portfolio weights"""
        w_T = (self.S_inv @ self.mu_e)/(np.ones(len(self.mu_e)) @ self.S_inv @ self.mu_e)
        return w_T
    
    def meanAndVariance(self,w):
        """For a given array of weight it computes the expected return and the variance"""
        mu_p = w @ self.mu_e + self.R_f
        var_p = w @ self.Sigma @ w
        return mu_p, var_p 
