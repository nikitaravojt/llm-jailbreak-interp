# Tracking LLM Refusal and Harmfulness Representations During Adversarial Jailbreak Optimisation

Code and experimental notebooks for my MSc Computer Science (Artificial Intelligence) dissertation at the University of Nottingham.

This project investigates how two internal safety-related representations in **Gemma-2-2B-it** — refusal and harmfulness — change throughout **Greedy Coordinate Gradient (GCG)** jailbreak optimisation.

The analysis tracks both signals across complete 500-step attack trajectories and compares their evolution with the model's behavioural outputs.

## Contents

The repository contains the Jupyter notebooks used during development and analysis. These are primarily **exploratory research notebooks rather than a polished software package**, so some repeated setup code, intermediate experiments and development checks are retained.

Saved notebook outputs are preserved where useful to provide a record of the experiments and intermediate results.

The main stages of the project include:

- Gemma-2-2B-it and GCG setup
- harmful and harmless prompt preparation
- refusal-direction extraction and causal validation
- harmfulness-direction extraction and causal validation
- projection calibration and decision boundaries
- GCG trajectory generation and replay
- refusal and harmfulness measurement throughout optimisation
- suffix-matched harmless baseline correction
- harmful-vs-harmless AUC analysis under adversarial suffixes
- behavioural classification of generated responses
- trajectory visualisation and aggregate analysis

## Main tools

The experiments primarily use:

- Python
- PyTorch
- Hugging Face Transformers
- TransformerLens
- nanoGCG
- NumPy / pandas
- scikit-learn
- matplotlib

Experiments were run using GPU-backed Kaggle and cloud environments.

## Dissertation

**Title:** *Tracking LLM Refusal and Harmfulness Representations During Adversarial Jailbreak Optimisation*

The dissertation should be treated as the authoritative description of the final methodology, experimental configuration and reported results. The notebooks additionally contain exploratory work and intermediate experiments conducted during development.

External datasets, libraries and prior implementations used in the project — including AdvBench, nanoGCG and the prompt datasets released with prior refusal-direction work — are cited in the dissertation.
