#!/usr/bin/env python3
"""
bayes_online.py
Simple online Bayesian update toolbox with examples:
- Beta-Bernoulli (discrete)
- Normal-Normal conjugate (known variance)
- Discrete Dynamic Bayes (HMM forward filtering)
- Linear Gaussian dynamic Bayes (Kalman filter)

Usage: python bayes_online.py
"""
import numpy as np

# -------------------------
# Stage 0 & 1: Priors & static Bayesian updates
# -------------------------
class BetaBernoulli:
    """Beta-Bernoulli conjugate model for binary observations.
    Prior: Beta(alpha, beta)
    Likelihood: Bernoulli(p)
    Online update: alpha += x, beta += 1-x
    """
    def __init__(self, alpha=1.0, beta=1.0):
        self.alpha = float(alpha)
        self.beta = float(beta)

    def update(self, x):
        """x in {0,1}"""
        if x not in (0,1):
            raise ValueError("Observation must be 0 or 1")
        self.alpha += x
        self.beta += 1 - x

    def posterior_mean(self):
        return self.alpha / (self.alpha + self.beta)

    def posterior_variance(self):
        a, b = self.alpha, self.beta
        return (a * b) / ((a + b)**2 * (a + b + 1))

    def predictive(self):
        # P(next=1) = alpha / (alpha + beta)
        return self.posterior_mean()

class NormalConjugate:
    """Normal-Normal conjugate with known observation variance sigma2.
    Prior: mu ~ N(mu0, sigma0^2)
    Likelihood: x ~ N(mu, sigma2)
    Online update formulas for posterior mean and variance.
    """
    def __init__(self, mu0=0.0, sigma0_sq=1.0, obs_sigma_sq=1.0):
        self.mu = float(mu0)
        self.sigma_sq = float(sigma0_sq)
        self.obs_sigma_sq = float(obs_sigma_sq)
        # store precision for convenience
        self._prec = 1.0 / self.sigma_sq

    def update(self, x):
        # posterior precision = prior_prec + 1/obs_var
        lik_prec = 1.0 / self.obs_sigma_sq
        post_prec = self._prec + lik_prec
        post_mu = (self.mu * self._prec + x * lik_prec) / post_prec
        # set new prior to posterior (online)
        self.mu = post_mu
        self.sigma_sq = 1.0 / post_prec
        self._prec = post_prec

    def posterior(self):
        return self.mu, self.sigma_sq

# -------------------------
# Stage 2: Dynamic Bayes - discrete HMM forward filter
# -------------------------
class DiscreteHMMFilter:
    """Simple forward filtering for a discrete-state HMM.
    - states: 0..K-1
    - transition: KxK matrix (row-stochastic or column? we'll use matrix T where
      T[i,j] = P(next_state=j | current_state=i))
    - emission_likelihood(state, observation) -> P(obs | state)
    """
    def __init__(self, init_state_dist, transition_matrix, emission_likelihood_fn):
        self.pi = np.asarray(init_state_dist, dtype=float)
        self.pi /= self.pi.sum()
        self.T = np.asarray(transition_matrix, dtype=float)
        assert self.T.shape[0] == self.T.shape[1] == len(self.pi)
        self.K = len(self.pi)
        self.emission_likelihood = emission_likelihood_fn

    def predict(self):
        # prior for next step: pi_pred = pi @ T
        return self.pi.dot(self.T)

    def update(self, observation):
        # predict
        prior = self.predict()
        # compute likelihood per state
        lik = np.array([self.emission_likelihood(s, observation) for s in range(self.K)], dtype=float)
        posterior_unnorm = prior * lik
        if posterior_unnorm.sum() == 0:
            # avoid division by zero: keep prior
            self.pi = prior
        else:
            self.pi = posterior_unnorm / posterior_unnorm.sum()
        return self.pi

# -------------------------
# Stage 2 (continuous): Kalman Filter (linear Gaussian dynamic)
# -------------------------
class KalmanFilter:
    """1D Kalman filter (extendable to vectors if needed).
    Model:
      x_{t} = A * x_{t-1} + q_t,  q_t ~ N(0, Q)
      y_{t} = H * x_t + r_t,      r_t ~ N(0, R)
    We keep scalar A,H,Q,R for simplicity.
    """
    def __init__(self, x0=0.0, P0=1.0, A=1.0, H=1.0, Q=1e-3, R=1.0):
        self.x = float(x0)
        self.P = float(P0)
        self.A = float(A)
        self.H = float(H)
        self.Q = float(Q)
        self.R = float(R)

    def predict(self):
        # x_prior = A x
        self.x = self.A * self.x
        self.P = self.A * self.P * self.A + self.Q
        return self.x, self.P

    def update(self, y):
        # predict first (typical KF predict-update cycle)
        x_prior, P_prior = self.predict()
        # Kalman gain
        S = self.H * P_prior * self.H + self.R
        K = (P_prior * self.H) / S
        # update
        self.x = x_prior + K * (y - self.H * x_prior)
        self.P = (1 - K * self.H) * P_prior
        return self.x, self.P

# -------------------------
# Stage 3: Example workflows (online)
# -------------------------
def example_beta_bernoulli():
    print("Beta-Bernoulli online example (coin flips)")
    model = BetaBernoulli(alpha=2, beta=2)  # prior moderate
    flips = [1,0,1,1,0,1,1,1,0]  # incoming observations
    for i,x in enumerate(flips,1):
        model.update(x)
        print(f"t={i} obs={x} posterior mean p={model.posterior_mean():.3f} var={model.posterior_variance():.4f}")

def example_normal_online():
    print("\nNormal-Normal online example (known obs variance=2.0)")
    model = NormalConjugate(mu0=0.0, sigma0_sq=5.0, obs_sigma_sq=2.0)
    data = [1.2, 0.9, 1.5, 2.0, 1.0]
    for i,x in enumerate(data,1):
        model.update(x)
        mu, var = model.posterior()
        print(f"t={i} obs={x:.2f} post_mu={mu:.3f} post_sigma={np.sqrt(var):.3f}")

def example_discrete_hmm():
    print("\nDiscrete HMM forward filtering example")
    # two-state example: states = {0: 'good', 1: 'bad'}
    pi0 = [0.8, 0.2]
    T = [[0.9, 0.1],
         [0.2, 0.8]]
    def lik(s, obs):
        # obs is boolean: True if symptom present
        if s == 0:  # good state: low prob of symptom
            return 0.1 if obs else 0.9
        else:
            return 0.8 if obs else 0.2
    f = DiscreteHMMFilter(pi0, T, lik)
    observations = [False, True, True, False, True]
    for t,obs in enumerate(observations,1):
        post = f.update(obs)
        print(f"t={t} obs={obs} posterior={post}")

def example_kalman():
    print("\nKalman filter example (1D tracking)")
    kf = KalmanFilter(x0=0.0, P0=1.0, A=1.0, H=1.0, Q=0.1, R=0.5)
    true_x = 0.0
    np.random.seed(0)
    for t in range(1,8):
        # simulate true state
        true_x = 1.0 * true_x + np.random.normal(0, np.sqrt(kf.Q))
        y = true_x + np.random.normal(0, np.sqrt(kf.R))
        x_est, P = kf.update(y)
        print(f"t={t} y={y:.3f} x_est={x_est:.3f} P={P:.3f} true_x={true_x:.3f}")

if __name__ == "__main__":
    example_beta_bernoulli()
    example_normal_online()
    example_discrete_hmm()
    example_kalman()
