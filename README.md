# Exposure-Aware Generative Creator Ranking

Implementation of **REDS (Relevance and Exposure Distribution Shaping)**, a framework for generative creator ranking that combines ranking relevance with exposure-aware regularization.

## Overview

The task is formulated as generative ranking, where a causal language model generates an ordered sequence of creator identifiers conditioned on a project prompt.

REDS introduces regularization objectives that shape the model's creator-token probabilities based on project relevance and historical attention.

The framework considers exposure allocation at two levels: across visibility groups and among individual creators.

## Methods

| Method | Identifier | Objective |
|---|---|---|
| NTP | `ntp` | Next-token prediction |
| FairGroup | `fairgroup` | NTP + relevance-balanced group KL |
| SuppGroup | `suppgroup` | NTP + high-attention suppressed group KL |
| SuppGroup+Creator | `suppgroup_creator` | NTP + group KL + creator KL |

## Implementation

The training objective combines next-token prediction with optional exposure regularization.

```text
L = lambda_ntp * L_NTP
  + lambda_kl * L_group_KL
  + lambda_kl_creator * L_creator_KL
```

The active losses depend on the selected method.

- For `ntp`, only the NTP loss is active.
- For `fairgroup` and `suppgroup`, the model uses NTP plus group-level KL.
- For `suppgroup_creator`, the model uses NTP, group-level KL, and creator-level KL.

The implementation uses two loss types, `ntp` and `ntp_kl`, with the following strategies.

| Component | Strategy |
|---|---|
| Balanced group target | `fair_group` |
| Suppressed group target | `high_suppressed_group` |
| Creator-level target | `creator_anti_attention` |

The exposure-aware losses operate on project-local creator-token probabilities during training.

## Training and Evaluation

**Training entry point**

`train/reranker/train_reranker.py`

**Evaluation entry point**

`infer/evaluate_reranker.py`

**Dependencies**

```bash
pip install -r requirements.txt
```

### Evaluation Metrics

| Category | Measures |
|---|---|
| Ranking relevance | Graded NDCG |
| Exposure allocation | Group exposure shares and exposure gap |
| Historical attention | Average and discounted attention |
| Creator concentration | Within-group Gini |

The methods are evaluated relative to the standard NTP baseline under comparable model settings. Exposure gap is defined only for methods with an explicit group target. Skill coverage is available as a separate helper and is not integrated into the main evaluator.

This repository contains the core REDS methods and the NTP baseline. The supplied configurations are examples; they are not a complete set of the reported experimental settings.
