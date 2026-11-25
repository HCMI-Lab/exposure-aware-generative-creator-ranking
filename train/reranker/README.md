# Generative Creator Ranking

This directory contains the causal language model training pipeline for REDS.

The dataset builder, training code, and evaluation code use prior attention scores and visibility groups consistently. The example configurations cover NTP, FairGroup, SuppGroup, and SuppGroup+Creator.

The exposure regularizers operate on project-local creator probabilities at the first creator-generation step. The creator-level objective uses a single target distribution over the project candidate set.

See the top-level README for the method definitions.
