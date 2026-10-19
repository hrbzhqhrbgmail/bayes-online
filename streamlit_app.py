import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

# Import models from bayes_online.py
from bayes_online import BetaBernoulli, NormalConjugate, DiscreteHMMFilter, KalmanFilter

st.set_page_config(page_title="Bayes Online — Interactive Demo", layout="wide")

st.title("Bayesian Posterior Online Iteration — Interactive Demo")
st.markdown("""
This interactive demo lets you play with the four example models included in the repository:

- Beta-Bernoulli (binary online updates)
- Normal-Normal conjugate (known observation variance)
- Discrete HMM forward filter (2-state example)
- 1D Kalman filter (linear Gaussian)

Use the controls to set priors and model parameters, then simulate observations step-by-step or run a small stream.

""")

MODEL_OPTIONS = ["Beta-Bernoulli", "Normal (conjugate)", "Discrete HMM", "Kalman Filter"]
model_choice = st.sidebar.selectbox("Choose model", MODEL_OPTIONS)

# session state to hold model and history
if "bayes_demo" not in st.session_state:
    st.session_state["bayes_demo"] = {}

state = st.session_state["bayes_demo"]

# Utilities
def st_line_plot(xs, ys, ylabel=""):
    fig = Figure(figsize=(6, 3))
    ax = fig.subplots()
    ax.plot(xs, ys, marker="o")
    ax.set_xlabel("t")
    ax.set_ylabel(ylabel)
    ax.grid(True)
    st.pyplot(fig)

# Beta-Bernoulli UI and logic
if model_choice == "Beta-Bernoulli":
    st.header("Beta-Bernoulli (online)")
    col1, col2 = st.columns(2)
    with col1:
        alpha0 = st.number_input("Prior alpha", min_value=0.0, value=2.0, step=1.0)
        beta0 = st.number_input("Prior beta", min_value=0.0, value=2.0, step=1.0)
        true_p = st.slider("True underlying p (for simulation)", 0.0, 1.0, 0.7, 0.01)
        sim_mode = st.selectbox("Observation source", ["Simulate", "Manual input"])
        if sim_mode == "Manual input":
            obs = st.selectbox("Next observation", [0, 1])
    with col2:
        if st.button("Reset model") or "model_type" not in state or state.get("model_type") != "beta":
            state.clear()
            state["model_type"] = "beta"
            state["model"] = BetaBernoulli(alpha=alpha0, beta=beta0)
            state["history"] = {"obs": [], "post_mean": []}

        st.markdown("**Posterior**")
        st.write(state["model"].alpha, state["model"].beta)
        st.write(f"Mean = {state['model'].posterior_mean():.4f}")
        st.write(f"Var = {state['model'].posterior_variance():.6f}")

        if st.button("Step: add one observation"):
            if sim_mode == "Simulate":
                o = int(np.random.rand() < true_p)
            else:
                o = int(obs)
            state["model"].update(o)
            state["history"]["obs"].append(o)
            state["history"]["post_mean"].append(state["model"].posterior_mean())

    st.subheader("History")
    if len(state["history"]["obs"]) > 0:
        st.write("Observations:", state["history"]["obs"]) 
        st_line_plot(range(1, len(state["history"]["post_mean"]) + 1), state["history"]["post_mean"], "posterior mean")

# Normal conjugate
if model_choice == "Normal (conjugate)":
    st.header("Normal-Normal conjugate (known observation variance)")
    col1, col2 = st.columns(2)
    with col1:
        mu0 = st.number_input("Prior mu0", value=0.0)
        sigma0_sq = st.number_input("Prior sigma0^2", value=5.0, min_value=1e-6)
        obs_sigma_sq = st.number_input("Observation variance (sigma^2)", value=2.0, min_value=1e-6)
        sim_mode = st.selectbox("Observation source", ["Simulate", "Manual input"])        
        if sim_mode == "Manual input":
            obs = st.number_input("Next observation value", value=0.0, step=0.1)
        true_mu = st.number_input("True underlying mu (for simulation)", value=1.0)
    with col2:
        if st.button("Reset model") or "model_type" not in state or state.get("model_type") != "normal":
            state.clear()
            state["model_type"] = "normal"
            state["model"] = NormalConjugate(mu0=mu0, sigma0_sq=sigma0_sq, obs_sigma_sq=obs_sigma_sq)
            state["history"] = {"obs": [], "post_mu": [], "post_sigma": []}

        mu, var = state["model"].posterior()
        st.markdown("**Posterior**")
        st.write(f"mu = {mu:.4f}")
        st.write(f"sigma = {np.sqrt(var):.4f}")

        if st.button("Step: add one observation"):
            if sim_mode == "Simulate":
                o = np.random.normal(true_mu, np.sqrt(obs_sigma_sq))
            else:
                o = float(obs)
            state["model"].update(o)
            mu, var = state["model"].posterior()
            state["history"]["obs"].append(o)
            state["history"]["post_mu"].append(mu)
            state["history"]["post_sigma"].append(np.sqrt(var))

    st.subheader("History")
    if len(state["history"]["obs"]) > 0:
        st.write("Observations:", np.round(state["history"]["obs"], 3).tolist())
        st_line_plot(range(1, len(state["history"]["post_mu"]) + 1), state["history"]["post_mu"], "posterior mu")

# Discrete HMM
if model_choice == "Discrete HMM":
    st.header("Discrete HMM forward filter (2 states)")
    col1, col2 = st.columns(2)
    with col1:
        p00 = st.slider("P(0->0)", 0.0, 1.0, 0.9)
        p11 = st.slider("P(1->1)", 0.0, 1.0, 0.8)
        pi0 = st.slider("Init P(state=0)", 0.0, 1.0, 0.8)
        sim_mode = st.selectbox("Observation source", ["Simulate", "Manual input"])        
        if sim_mode == "Manual input":
            obs = st.selectbox("Next observation (symptom present)", [False, True])
    with col2:
        if st.button("Reset model") or "model_type" not in state or state.get("model_type") != "hmm":
            T = [[p00, 1-p00], [1-p11, p11]]
            def lik(s, obs):
                if s == 0:
                    return 0.1 if obs else 0.9
                else:
                    return 0.8 if obs else 0.2
            state.clear()
            state["model_type"] = "hmm"
            state["model"] = DiscreteHMMFilter([pi0, 1-pi0], T, lik)
            state["history"] = {"obs": [], "post": []}

        st.markdown("**Posterior state dist**")
        st.write(state["model"].pi)

        if st.button("Step: add one observation"):
            if sim_mode == "Simulate":
                # simulate from marginal predictive (approx)
                prior = state["model"].predict()
                # sample state then obs
                s = np.random.choice([0,1], p=prior)
                o = np.random.rand() < (0.1 if s==0 else 0.8)
            else:
                o = bool(obs)
            post = state["model"].update(o)
            state["history"]["obs"].append(bool(o))
            state["history"]["post"].append(post.copy())

    st.subheader("History")
    if len(state["history"]["obs"]) > 0:
        st.write("Observations:", state["history"]["obs"]) 
        # plot P(state=0)
        p0 = [p[0] for p in state["history"]["post"]]
        st_line_plot(range(1, len(p0) + 1), p0, "P(state=0)")

# Kalman Filter
if model_choice == "Kalman Filter":
    st.header("1D Kalman filter")
    col1, col2 = st.columns(2)
    with col1:
        x0 = st.number_input("Initial x0", value=0.0)
        P0 = st.number_input("Initial P0", value=1.0, min_value=1e-6)
        A = st.number_input("A (state trans)", value=1.0)
        Q = st.number_input("Q (process noise)", value=0.1, min_value=0.0)
        R = st.number_input("R (obs noise)", value=0.5, min_value=1e-6)
        sim_mode = st.selectbox("Observation source", ["Simulate", "Manual input"])        
        if sim_mode == "Manual input":
            obs = st.number_input("Next observation y", value=0.0)
        true_x = st.number_input("True initial x (for sim)", value=0.0)
    with col2:
        if st.button("Reset model") or "model_type" not in state or state.get("model_type") != "kf":
            state.clear()
            state["model_type"] = "kf"
            state["model"] = KalmanFilter(x0=x0, P0=P0, A=A, H=1.0, Q=Q, R=R)
            state["history"] = {"y": [], "x_est": [], "P": []}
            # store true state for simulation
            state["true_x"] = float(true_x)

        st.markdown("**Posterior**")
        st.write(f"x_est = {state['model'].x:.4f}")
        st.write(f"P = {state['model'].P:.4f}")

        if st.button("Step: add one observation"):
            if sim_mode == "Simulate":
                # evolve true state and make noisy observation
                state["true_x"] = A * state["true_x"] + np.random.normal(0, np.sqrt(Q))
                y = state["true_x"] + np.random.normal(0, np.sqrt(R))
            else:
                y = float(obs)
            x_est, P = state["model"].update(y)
            state["history"]["y"].append(y)
            state["history"]["x_est"].append(x_est)
            state["history"]["P"].append(P)

    st.subheader("History")
    if len(state["history"]["y"]) > 0:
        st.write("Observations y:", np.round(state["history"]["y"], 3).tolist())
        st_line_plot(range(1, len(state["history"]["x_est"]) + 1), state["history"]["x_est"], "x_est")

st.sidebar.markdown("---")
st.sidebar.markdown("Run notes: use Reset when you change prior parameters to reinitialize the model. Use Step to simulate or input the next observation.")
