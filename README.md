# bayes-online

A small teaching project demonstrating Bayesian posterior updates from static priors to online dynamic inference. This repository contains simple, readable implementations suitable for learning and quick experimentation.

Contents
- bayes_online.py — single-file implementation with four stages:
  - Stage 0: prior representations (Beta, Normal)
  - Stage 1: static Bayesian updates (Beta-Bernoulli, Normal-Normal conjugacy)
  - Stage 2: dynamic Bayesian models (discrete HMM forward filter, 1D Kalman filter)
  - Stage 3: online posterior updates (each class supports incremental updates)
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

3. Run the examples:

   python bayes_online.py

This will print step-by-step online updates for the example models.

License

This project is licensed under the MIT License — see LICENSE for details.
