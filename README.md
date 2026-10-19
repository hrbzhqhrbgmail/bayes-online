# bayes-online

A small teaching project demonstrating Bayesian posterior updates from static priors to online dynamic inference. This repository contains simple, readable implementations suitable for learning and quick experimentation.

This repository now includes an interactive Streamlit demo (streamlit_app.py) to step through online updates and visualize posterior trajectories.

Contents
- bayes_online.py — single-file implementation with four stages:
  - Stage 0: prior representations (Beta, Normal)
  - Stage 1: static Bayesian updates (Beta-Bernoulli, Normal-Normal conjugacy)
  - Stage 2: dynamic Bayesian models (discrete HMM forward filter, 1D Kalman filter)
  - Stage 3: online posterior updates (each class supports incremental updates)
- streamlit_app.py — interactive Streamlit demo
- requirements.txt — Python dependencies
- LICENSE — MIT license
- .gitignore — Python gitignore

Quickstart

1. Clone the repo:

   git clone https://github.com/hrbzhqhrbgmail/bayes-online.git
   cd bayes-online

2. Create a virtualenv and install dependencies:

   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt

3. Run the examples (console):

   python bayes_online.py

4. Run the interactive Streamlit demo:

   streamlit run streamlit_app.py

License

This project is licensed under the MIT License — see LICENSE for details.
