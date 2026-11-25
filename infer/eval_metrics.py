import os
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json
import pandas as pd
import pytrec_eval
import math
import re


GROUP_NAMES = ("low", "mid", "high")
PAPER_RESULT_COLUMNS = (
    "graded_ndcg_at_10",
    "graded_ndcg_at_20",
    "avg_prior_attention_at_10",
    "avg_prior_attention_at_20",
    "disc_prior_attention_at_10",
    "disc_prior_attention_at_20",
    "exp_share_low_at_10",
    "exp_share_mid_at_10",
    "exp_share_high_at_10",
    "dexp_share_low_at_10",
    "dexp_share_mid_at_10",
    "dexp_share_high_at_10",
    "dgroup_gap_at_10",
    "within_group_gini_low_at_10",
    "within_group_gini_mid_at_10",
    "within_group_gini_high_at_10",
)


def _rank_exposure(rank: int) -> float:
    """Position exposure for 1-indexed ranks."""
    return 1.0 / math.log2(rank + 1)

# -----------------------------
# Measures
# -----------------------------
def make_measures_for_ks(ks=(2, 5, 10)) -> set:
    """
    Measures that match your current reporting needs.
    make_measures_for_ks defines TREC-native metrics that:

    are computed by pytrec_eval

    must be known to pytrec_eval.RelevanceEvaluator
    """
    measures = set()
    for k in ks:
        measures |= {f"P_{k}", f"recall_{k}", f"ndcg_cut_{k}", f"map_cut_{k}",}
    return measures


def evaluate_once(
    qrels: Dict[str, Dict[str, int]],
    run: Dict[str, Dict[str, float]],
    ks=(2, 5, 10),
    extra_measures: Optional[List[str]] = None,
) -> Tuple[set, Dict[str, Dict[str, float]]]:
    measures = make_measures_for_ks(ks)
    if extra_measures:
        measures |= set(extra_measures)

    evaluator = pytrec_eval.RelevanceEvaluator(qrels, measures)
    per_query = evaluator.evaluate(run)
    return measures, per_query

def evaluate_binary_and_graded(
    qrels_binary: Dict[str, Dict[str, int]],
    qrels_graded: Dict[str, Dict[str, int]],
    run: Dict[str, Dict[str, float]],
    ks=(2, 5, 10),
    extra_measures: Optional[List[str]] = None,
):
    """
    Evaluate binary and graded metrics separately and keep separate names.

    Binary side:
      - P@k
      - Recall@k
      - MAP@k
      - binary NDCG@k

    Graded side:
      - graded NDCG@k

    Note:
      P / Recall are inherently binary-style metrics.
      MAP is usually interpreted in a binary way too.
      The main metric that meaningfully differs here is NDCG.
    """
    binary_measures = set()
    graded_measures = set()

    for k in ks:
        binary_measures |= {
            f"P_{k}",
            f"recall_{k}",
            f"map_cut_{k}",
            f"ndcg_cut_{k}",   # binary NDCG
        }
        graded_measures |= {
            f"ndcg_cut_{k}",   # graded NDCG
        }

    if extra_measures:
        binary_measures |= set(extra_measures)

    per_query_binary_raw = pytrec_eval.RelevanceEvaluator(
        qrels_binary, binary_measures
    ).evaluate(run)

    per_query_graded_raw = pytrec_eval.RelevanceEvaluator(
        qrels_graded, graded_measures
    ).evaluate(run)

    # Merge with explicit names
    per_query = {}
    all_qids = set(per_query_binary_raw) | set(per_query_graded_raw)

    for qid in all_qids:
        per_query[qid] = {}

        if qid in per_query_binary_raw:
            for metric_name, value in per_query_binary_raw[qid].items():
                if metric_name.startswith("ndcg_cut_"):
                    per_query[qid][f"binary_{metric_name}"] = value
                else:
                    per_query[qid][metric_name] = value

        if qid in per_query_graded_raw:
            for metric_name, value in per_query_graded_raw[qid].items():
                if metric_name.startswith("ndcg_cut_"):
                    per_query[qid][f"graded_{metric_name}"] = value
                else:
                    per_query[qid][f"graded_{metric_name}"] = value

    measures = set()

    for k in ks:
        measures |= {
            f"P_{k}",
            f"recall_{k}",
            f"map_cut_{k}",
            f"binary_ndcg_cut_{k}",
            f"graded_ndcg_cut_{k}",
        }

    if extra_measures:
        measures |= set(extra_measures)

    return measures, per_query
    
def aggregate_metrics(per_query: Dict[str, Dict[str, float]], measures: set) -> dict:
    agg = {}
    for m in measures:
        vals = [metrics[m] for metrics in per_query.values() if m in metrics]
        if m.startswith(("within_group_gini_", "dexp_share_", "dgroup_gap_")):
            # Undefined concentration, discounted exposure, or target gap is
            # missing, not zero. Average only defined observations.
            vals = [value for value in vals if value is not None and math.isfinite(value)]
            agg[m] = float(sum(vals) / len(vals)) if vals else float("nan")
        else:
            agg[m] = float(sum(vals) / len(vals)) if vals else 0.0
    agg["num_q"] = float(len(per_query))
    return agg

def compute_prior_attention_metrics(run, creator_prior_attention, ks):
    """Average and rank-discounted prior attention with log1p transformation."""
    avg = {}
    disc = {}
    if creator_prior_attention is None:
        return {}, {}

    for k in ks:
        exp_vals = []
        dexp_vals = []

        for qid, docs in run.items():
            ranked = sorted(docs.items(), key=lambda x: x[1], reverse=True)[:k]
            a_vals = [float(creator_prior_attention.get(docid, 0.0)) for docid, _ in ranked]

            # If fewer than k docs exist, pad with zeros to keep 1/k semantics
            if len(a_vals) < k:
                a_vals += [0.0] * (k - len(a_vals))

            # Exp@k = g( (1/k) * sum a_i )
            mean_a = sum(a_vals) / k
            exp_vals.append(math.log1p(mean_a))

            # DExp@k = g( sum a_i / log2(i+1) )
            disc_sum = sum(
                a / math.log2(i + 2) for i, a in enumerate(a_vals)  # i=0 -> log2(2)=1
            )
            dexp_vals.append(math.log1p(disc_sum))

        avg[f"AvgPriorAttention@{k}"] = sum(exp_vals) / len(exp_vals) if exp_vals else 0.0
        disc[f"DiscPriorAttention@{k}"] = sum(dexp_vals) / len(dexp_vals) if dexp_vals else 0.0

    return avg, disc


def add_prior_attention_metrics_to_agg(agg, run, creator_prior_attention, ks):
    """Compute prior attention metrics without modifying ranking scores."""
    avg, disc = compute_prior_attention_metrics(run, creator_prior_attention, ks)
    agg.update(avg)
    agg.update(disc)
    return agg

def gini(values) -> Optional[float]:
    """Return Gini for non-negative exposures, or None when undefined."""
    vals = sorted(float(value) for value in values)
    n = len(vals)
    if n == 0:
        return None
    total = sum(vals)
    if total <= 0.0:
        return None
    weighted_sum = sum((idx + 1) * value for idx, value in enumerate(vals))
    return (2.0 * weighted_sum) / (n * total) - (n + 1.0) / n


def _normalize_group_id(raw_group_id) -> Optional[int]:
    if raw_group_id is None:
        return None
    if isinstance(raw_group_id, str):
        lowered = raw_group_id.strip().lower()
        if lowered in GROUP_NAMES:
            return GROUP_NAMES.index(lowered)
    try:
        group_id = int(raw_group_id)
    except (TypeError, ValueError):
        return None
    if 0 <= group_id < len(GROUP_NAMES):
        return group_id
    return None


def compute_group_exposure_shares_at_k(
    run: Dict[str, Dict[str, float]],
    qid_to_group_ids: Dict[str, Dict[str, int]],
    k: int = 10,
) -> Dict[str, Dict[str, float]]:
    """Count group representation at cutoff k; unfilled ranks contribute zero."""
    if k <= 0:
        raise ValueError("k must be positive")
    out: Dict[str, Dict[str, float]] = {}
    for qid, docscores in run.items():
        group_map = qid_to_group_ids.get(qid, {})
        if not group_map:
            out[qid] = {}
            continue

        ranked = sorted(docscores.items(), key=lambda x: x[1], reverse=True)[:k]
        if any(_normalize_group_id(group_map.get(str(docid))) is None for docid, _ in ranked):
            out[qid] = {}
            continue

        group_counts = {group_name: 0.0 for group_name in GROUP_NAMES}

        for docid, _ in ranked:
            group_id = _normalize_group_id(group_map.get(str(docid)))
            group_counts[GROUP_NAMES[group_id]] += 1.0

        out[qid] = {
            f"exp_share_{group_name}_at_{k}": (
                group_counts[group_name] / k
            )
            for group_name in GROUP_NAMES
        }
    return out


def compute_discounted_group_exposure_shares_at_k(
    run: Dict[str, Dict[str, float]],
    qid_to_group_ids: Dict[str, Dict[str, int]],
    k: int = 10,
) -> Dict[str, Dict[str, float]]:
    """Normalize by all k rank weights; unfilled positions contribute zero."""
    if k <= 0:
        raise ValueError("k must be positive")
    norm = sum(_rank_exposure(rank) for rank in range(1, k + 1))
    out: Dict[str, Dict[str, float]] = {}
    for qid, docscores in run.items():
        group_map = qid_to_group_ids.get(qid, {})
        if not group_map:
            out[qid] = {}
            continue

        ranked = sorted(docscores.items(), key=lambda x: x[1], reverse=True)[:k]
        if any(_normalize_group_id(group_map.get(str(docid))) is None for docid, _ in ranked):
            out[qid] = {}
            continue

        group_scores = {group_name: 0.0 for group_name in GROUP_NAMES}

        for rank_idx, (docid, _) in enumerate(ranked, start=1):
            discount = _rank_exposure(rank_idx)
            group_id = _normalize_group_id(group_map.get(str(docid)))
            group_scores[GROUP_NAMES[group_id]] += discount

        out[qid] = {
            f"dexp_share_{group_name}_at_{k}": (
                group_scores[group_name] / norm
            )
            for group_name in GROUP_NAMES
        }
    return out


def compute_within_group_concentration_metrics(
    run: Dict[str, Dict[str, float]],
    qid_to_group_ids: Dict[str, Dict[str, int]],
    ks: Tuple[int, ...],
) -> Dict[str, Dict[str, float]]:
    """
    Compute intra-group creator-level exposure concentration.

    Each visibility group includes all candidate creators for the query. Selected
    top-k creators receive rank-position exposure, while unselected candidates
    receive zero exposure.
    """
    out: Dict[str, Dict[str, float]] = {}
    for qid, docscores in run.items():
        group_map = qid_to_group_ids.get(qid, {})
        if not group_map:
            out[qid] = {}
            continue

        ranked = sorted(docscores.items(), key=lambda x: x[1], reverse=True)
        query_metrics: Dict[str, float] = {}

        for k in ks:
            ranked_topk = ranked[:k]
            if any(_normalize_group_id(group_map.get(str(docid))) is None for docid, _ in ranked_topk):
                continue

            selected_exposure = {
                str(docid): _rank_exposure(rank_idx)
                for rank_idx, (docid, _) in enumerate(ranked_topk, start=1)
            }

            for group_id, group_name in enumerate(GROUP_NAMES):
                exposures = [
                    selected_exposure.get(str(candidate_id), 0.0)
                    for candidate_id, raw_group_id in group_map.items()
                    if _normalize_group_id(raw_group_id) == group_id
                ]
                value = gini(exposures)
                if value is not None:
                    query_metrics[f"within_group_gini_{group_name}_at_{k}"] = value

        out[qid] = query_metrics
    return out


def compute_target_group_distribution(
    *,
    label_map: Dict[str, float],
    group_map: Dict[str, int],
    kl_target_strategy: str,
    alpha_fair: float,
    delta_high: float,
) -> Optional[Dict[str, float]]:
    """Build the same relevance-based target as the training group KL.

    Baselines without a declared target have no defined target gap.
    """
    if kl_target_strategy in (None, "none", "unknown"):
        return None
    if kl_target_strategy not in ("fair_group", "high_suppressed_group"):
        raise ValueError(f"Unsupported group target strategy: {kl_target_strategy!r}")

    if any(_normalize_group_id(group_map.get(str(docid))) is None for docid in label_map):
        return None

    group_mass = {group_name: 0.0 for group_name in GROUP_NAMES}
    for docid, label in label_map.items():
        relevance = float(label)
        if not math.isfinite(relevance) or relevance < 0.0:
            raise ValueError("Group target requires finite, non-negative relevance labels.")
        group_id = _normalize_group_id(group_map.get(str(docid)))
        group_mass[GROUP_NAMES[group_id]] += relevance

    total_relevance_mass = sum(group_mass.values())
    if total_relevance_mass <= 0:
        return None
    epsilon_rel = {
        group_name: group_mass[group_name] / total_relevance_mass
        for group_name in GROUP_NAMES
    }

    if kl_target_strategy == "fair_group":
        alpha = float(alpha_fair)
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha_fair must be between zero and one.")
        uniform_mass = 1.0 / len(GROUP_NAMES)
        return {
            group_name: (1.0 - alpha) * epsilon_rel[group_name] + alpha * uniform_mass
            for group_name in GROUP_NAMES
        }

    delta = float(delta_high)
    if not math.isfinite(delta) or delta < 0.0:
        raise ValueError("delta_high must be finite and non-negative.")
    remaining_high = max(epsilon_rel["high"] - delta, 0.0)
    removed_mass = epsilon_rel["high"] - remaining_high
    target = {
        group_name: (
            remaining_high if group_name == "high"
            else epsilon_rel[group_name] + removed_mass / (len(GROUP_NAMES) - 1)
        )
        for group_name in GROUP_NAMES
    }
    target_sum = sum(target.values())
    return {group_name: value / target_sum for group_name, value in target.items()}


def compute_discounted_group_gap_at_k(
    run: Dict[str, Dict[str, float]],
    qid_to_labels: Dict[str, Dict[str, float]],
    qid_to_group_ids: Dict[str, Dict[str, int]],
    *,
    k: int = 10,
    kl_target_strategy: str = "none",
    alpha_fair: float = 0.2,
    delta_high: float = 0.05,
) -> Dict[str, Dict[str, float]]:
    discounted_shares = compute_discounted_group_exposure_shares_at_k(
        run=run,
        qid_to_group_ids=qid_to_group_ids,
        k=k,
    )
    out: Dict[str, Dict[str, float]] = {}
    for qid, share_metrics in discounted_shares.items():
        label_map = qid_to_labels.get(qid)
        group_map = qid_to_group_ids.get(qid, {})
        # An empty metrics dictionary means group membership is missing or
        # invalid, not a valid zero-exposure ranking. Empty rankings with known
        # groups have explicit zero shares and can still have a defined gap.
        if not label_map or not share_metrics:
            out[qid] = {}
            continue

        target_distribution = compute_target_group_distribution(
            label_map=label_map,
            group_map=group_map,
            kl_target_strategy=kl_target_strategy,
            alpha_fair=alpha_fair,
            delta_high=delta_high,
        )
        if target_distribution is None:
            # Skip queries with no target mass rather than injecting an arbitrary value.
            out[qid] = {}
            continue

        gap = 0.0
        for group_name in GROUP_NAMES:
            gap += abs(
                float(share_metrics[f"dexp_share_{group_name}_at_{k}"])
                - float(target_distribution[group_name])
            )
        out[qid] = {f"dgroup_gap_at_{k}": gap}
    return out


def add_group_exposure_metrics_to_per_query(
    *,
    per_query: Dict[str, Dict[str, float]],
    run: Dict[str, Dict[str, float]],
    qid_to_labels: Dict[str, Dict[str, float]],
    qid_to_group_ids: Dict[str, Dict[str, int]],
    k: int = 10,
    kl_target_strategy: str = "none",
    alpha_fair: float = 0.2,
    delta_high: float = 0.05,
) -> Dict[str, Dict[str, float]]:
    plain = compute_group_exposure_shares_at_k(
        run=run,
        qid_to_group_ids=qid_to_group_ids,
        k=k,
    )
    discounted = compute_discounted_group_exposure_shares_at_k(
        run=run,
        qid_to_group_ids=qid_to_group_ids,
        k=k,
    )
    gap = compute_discounted_group_gap_at_k(
        run=run,
        qid_to_labels=qid_to_labels,
        qid_to_group_ids=qid_to_group_ids,
        k=k,
        kl_target_strategy=kl_target_strategy,
        alpha_fair=alpha_fair,
        delta_high=delta_high,
    )

    out: Dict[str, Dict[str, float]] = {}
    all_qids = set(plain) | set(discounted) | set(gap)
    for qid in all_qids:
        merged = {}
        merged.update(plain.get(qid, {}))
        merged.update(discounted.get(qid, {}))
        merged.update(gap.get(qid, {}))
        per_query.setdefault(qid, {}).update(merged)
        out[qid] = merged
    return out

# -----------------------------
# Saving CSVs
# -----------------------------
def save_metrics_csv(
    per_query: Dict[str, Dict[str, float]],
    model_name: str,
    meta: Optional[dict],
    output_csv: str,
    ks=(2, 5, 10),
) -> None:
    """
    Save per-query metrics table.
    """
    rows = []
    for qid, metrics in per_query.items():
        row = {
            "sample_id": qid,
            "model_name": model_name,
        }
        if meta:
            row.update(
                {
                "method_name": meta.get("method_name", "unknown"),
                "loss_type": meta.get("loss_type", "unknown"),
                "kl_target_strategy": meta.get("kl_target_strategy", "unknown"),
                "creator_kl_strategy": meta.get("creator_kl_strategy", "none"),
                "alpha_fair": meta.get("alpha_fair", 0.0),
                "delta_high": meta.get("delta_high", 0.0),
                "beta_prior_attention": meta.get("beta_prior_attention", 0.0),
                "tau_relevance": meta.get("tau_relevance", 1.0),
                }
            )

        # Creator ranking ranking metrics are computed over generated creator order.
        for k in ks:
            row[f"GradedNDCG@{k}"] = metrics.get(f"graded_ndcg_cut_{k}", 0.0)
            for group_name in GROUP_NAMES:
                row[f"WithinGroupGini{group_name.title()}@{k}"] = metrics.get(
                    f"within_group_gini_{group_name}_at_{k}",
                    float("nan"),
                )
        row["ExpShareLow@10"] = metrics.get("exp_share_low_at_10", 0.0)
        row["ExpShareMid@10"] = metrics.get("exp_share_mid_at_10", 0.0)
        row["ExpShareHigh@10"] = metrics.get("exp_share_high_at_10", 0.0)
        row["DExpShareLow@10"] = metrics.get("dexp_share_low_at_10", float("nan"))
        row["DExpShareMid@10"] = metrics.get("dexp_share_mid_at_10", float("nan"))
        row["DExpShareHigh@10"] = metrics.get("dexp_share_high_at_10", float("nan"))
        row["DGroupGap@10"] = metrics.get("dgroup_gap_at_10", float("nan"))

        if "recip_rank" in metrics:
            row["mrr"] = metrics.get("recip_rank", 0.0)

        rows.append(row)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"✅ Per-query metrics saved to {output_csv}")


def append_agg_metrics_csv(
    agg: dict,
    ks: tuple,
    output_csv: str,
    compare_with_baseline: bool = False,
) -> None:
    """
    Append one aggregated row per run to output_csv.
    
    """
    row = build_agg_metrics_row(
        agg=agg,
        ks=ks,
    )

    df_new = pd.DataFrame([row])

    if os.path.exists(output_csv):
        try:
            df_old = pd.read_csv(output_csv)
            required = {col for col in df_new.columns if "prior_attention" in col}
            if not df_old.empty and not required.issubset(df_old.columns):
                raise ValueError("Existing results CSV uses a different metric schema. Choose a new results filename or migrate the existing columns before appending.")
            df_old = df_old.reindex(columns=df_new.columns)
            df = pd.concat([df_old, df_new], ignore_index=True)
        except pd.errors.EmptyDataError:
            df = df_new
    else:
        df = df_new

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"📈 Results appended to {output_csv}")

    if compare_with_baseline:
        maybe_write_baseline_comparison_csv(
            current_row=row,
            output_csv=output_csv,
        )


def build_agg_metrics_row(
    agg: dict,
    ks: tuple,
) -> dict:
    """Build one aggregated evaluation row using the standard results schema."""
    row = {
        "run_time": agg.get("run_time", "unknown"),
        "model_name_or_path": agg.get("model_name_or_path", "unknown"),
        "model_dir": agg.get("model_dir", "unknown"),
        "method_name": agg.get("method_name", "unknown"),
        "loss_type": agg.get("loss_type", "unknown"),
        "adapter_mode": agg.get("adapter_mode", "unknown"),
        "num_q": agg.get("num_q", 0.0),
        "lambda_ntp": agg.get("lambda_ntp", 0.0),
        "lambda_kl": agg.get("lambda_kl", 0.0),
        "lambda_kl_creator": agg.get("lambda_kl_creator", 0.0),
        "kl_target_strategy": agg.get("kl_target_strategy", "unknown"),
        "creator_kl_strategy": agg.get("creator_kl_strategy", "none"),
        "alpha_fair": agg.get("alpha_fair", 0.0),
        "delta_high": agg.get("delta_high", 0.0),
        "beta_prior_attention": agg.get("beta_prior_attention", 0.0),
        "tau_relevance": agg.get("tau_relevance", 1.0),
    }

    for k in ks:
        row[f"graded_ndcg_at_{k}"] = agg.get(f"graded_ndcg_cut_{k}", 0.0)
    for k in ks:
        row[f"avg_prior_attention_at_{k}"] = agg.get(f"AvgPriorAttention@{k}", 0.0)
        row[f"disc_prior_attention_at_{k}"] = agg.get(f"DiscPriorAttention@{k}", 0.0)
    for group_name in GROUP_NAMES:
        row[f"within_group_gini_{group_name}_at_10"] = agg.get(
            f"within_group_gini_{group_name}_at_10",
            float("nan"),
        )
    row["exp_share_low_at_10"] = agg.get("exp_share_low_at_10", 0.0)
    row["exp_share_mid_at_10"] = agg.get("exp_share_mid_at_10", 0.0)
    row["exp_share_high_at_10"] = agg.get("exp_share_high_at_10", 0.0)
    row["dexp_share_low_at_10"] = agg.get("dexp_share_low_at_10", float("nan"))
    row["dexp_share_mid_at_10"] = agg.get("dexp_share_mid_at_10", float("nan"))
    row["dexp_share_high_at_10"] = agg.get("dexp_share_high_at_10", float("nan"))
    row["dgroup_gap_at_10"] = agg.get("dgroup_gap_at_10", float("nan"))
    return row


def _is_metric_column(column_name: str) -> bool:
    """Return whether a results column represents a metric rather than metadata."""
    name = str(column_name).lower()
    metric_prefixes = (
        "graded_ndcg_at_",
        "avg_prior_attention_at_",
        "disc_prior_attention_at_",
        "exp_share_",
        "dexp_share_",
        "dgroup_gap_",
        "within_group_gini_",
    )
    metric_exact = set()
    return name.startswith(metric_prefixes) or name in metric_exact


def _is_lower_better_metric(column_name: str) -> bool:
    """Return whether smaller values indicate better performance for this metric."""
    name = str(column_name).lower()
    return (
        name.startswith("avg_prior_attention_at_")
        or name.startswith("disc_prior_attention_at_")
        or "dgroup_gap" in name
        or name.startswith("within_group_gini_")
    )


def _infer_baseline_comparison_path(output_csv: str) -> Path:
    """Create the row-based baseline comparison path from the evaluation results filename."""
    output_path = Path(output_csv)
    return output_path.with_name(f"baseline_comparison_{output_path.name}")


def _load_latest_baseline_row(results_csv: Path, current_row: dict) -> Optional[dict]:
    """Load the latest plain NTP baseline row for the same model block."""
    if not results_csv.exists():
        return None
    try:
        baseline_df = pd.read_csv(results_csv)
    except pd.errors.EmptyDataError:
        return None
    if baseline_df.empty:
        return None
    candidates = baseline_df[
        (baseline_df["method_name"] == "ntp")
        & (baseline_df["model_name_or_path"] == current_row.get("model_name_or_path"))
    ]
    if "adapter_mode" in baseline_df.columns:
        candidates = candidates[
            candidates["adapter_mode"] == current_row.get("adapter_mode")
        ]
    if candidates.empty:
        return None
    return dict(candidates.iloc[-1].to_dict())


def maybe_write_baseline_comparison_csv(
    current_row: dict,
    output_csv: str,
) -> None:
    """
    Write or append a row-based baseline comparison CSV beside the main results CSV.

    Positive ``*_diff`` always means the current method is better than the baseline:
      - higher-is-better metrics: current - baseline
      - lower-is-better metrics: baseline - current
    """
    output_path = Path(output_csv)
    if output_path.name.startswith("baseline_comparison_"):
        return

    if current_row.get("method_name") == "ntp":
        return

    baseline_row = _load_latest_baseline_row(output_path, current_row)
    if baseline_row is None:
        print(
            "Warning: baseline comparison skipped because a matching NTP baseline "
            "for the same model was not found in the results CSV."
        )
        return

    comparison_row = dict(current_row)
    comparison_row["baseline_run_time"] = baseline_row.get("run_time", "unknown")

    metric_columns = [
        column_name
        for column_name in current_row.keys()
        if _is_metric_column(column_name)
    ]
    for column_name in metric_columns:
        # NTP has no training exposure target. Even legacy NTP CSVs may
        # contain a gap computed against an implicit, different target.
        baseline_value = (
            float("nan") if column_name.startswith("dgroup_gap_")
            else baseline_row.get(column_name)
        )
        comparison_row[f"{column_name}_baseline"] = baseline_value

        current_value = current_row.get(column_name)
        if pd.isna(current_value) or pd.isna(baseline_value):
            comparison_row[f"{column_name}_diff"] = float("nan")
            continue

        current_float = float(current_value)
        baseline_float = float(baseline_value)
        if _is_lower_better_metric(column_name):
            comparison_row[f"{column_name}_diff"] = baseline_float - current_float
        else:
            comparison_row[f"{column_name}_diff"] = current_float - baseline_float

    comparison_csv = _infer_baseline_comparison_path(output_csv)
    df_new = pd.DataFrame([comparison_row])
    if comparison_csv.exists():
        try:
            df_old = pd.read_csv(comparison_csv)
            all_columns = list(dict.fromkeys([*df_old.columns.tolist(), *df_new.columns.tolist()]))
            df = pd.concat(
                [df_old.reindex(columns=all_columns), df_new.reindex(columns=all_columns)],
                ignore_index=True,
            )
        except pd.errors.EmptyDataError:
            df = df_new
    else:
        df = df_new

    comparison_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(comparison_csv, index=False)
    print(f"📊 Baseline comparison appended to {comparison_csv}")


def infer_run_time_from_model_arg(model_arg: str) -> str:
    model_path = Path(model_arg)
    candidates = [model_path.name, model_path.parent.name]
    timestamp_pattern = re.compile(r"^\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}$")
    for candidate in candidates:
        if timestamp_pattern.match(candidate):
            return candidate
    return "unknown"

# -----------------------------
# Debug / sanity helpers
# -----------------------------
def analyze_run_binary(qrels: dict, run: dict) -> None:
    """
    Works with binary qrels (label>0 -> 1).
    Helpful sanity checks.
    """
    num_q = len(run)
    if num_q == 0:
        print("⚠️ No queries in run.")
        return

    docs_per_q = []
    rel_in_run = []
    nonrel_in_run = []
    zero_hit = 0

    for qid in run:
        retrieved = set(run[qid].keys())
        relevant = set(qrels.get(qid, {}).keys())

        docs_per_q.append(len(retrieved))
        rel_count = len(retrieved & relevant)
        nonrel_count = len(retrieved - relevant)

        rel_in_run.append(rel_count)
        nonrel_in_run.append(nonrel_count)

        if rel_count == 0:
            zero_hit += 1

    print("⚙️ RUN ANALYSIS (binary qrels)")
    print(f"  Queries evaluated         : {num_q}")
    print(f"  Avg docs / query          : {sum(docs_per_q)/num_q:.2f}")
    print(f"  Avg relevant retrieved    : {sum(rel_in_run)/num_q:.2f}")
    print(f"  Avg non-relevant retrieved: {sum(nonrel_in_run)/num_q:.2f}")
    print(f"  Queries with 0 relevant   : {zero_hit} ({100*zero_hit/num_q:.1f}%)")
    print("")
    

def load_run_config_or_default(model_arg: str) -> dict:
    """
    Load run_config.json from either:
      - <model_arg>/run_config.json
      - <model_arg>/../run_config.json   if model_arg points at a saved model dir

    If model_arg is not a local path or the file does not exist,
    return defaults.
    """
    model_path = Path(model_arg)

    defaults = {
        "method_name": "unknown",
        "loss_type": "unknown",
        "model_name_or_path": model_arg,
        "run_time": "unknown",
        "run_dir": str(model_path.parent if model_path.name == "model" else model_path),
        "adapter_mode": "unknown",
        "use_lora": False,
        "use_qlora": False,
        "lora_r": 0,
        "lora_alpha": 0,
        "lora_dropout": 0.0,
        "epochs": 0,
        "batch_size": 0,
        "gradient_accumulation_steps": 0,
        "lr": 0.0,
        "max_len": 2048,
        "max_creator_candidates": 64,
        "lambda_ntp": 0.0,
        "lambda_kl": 0.0,
        "lambda_kl_creator": 0.0,
        "kl_target_strategy": "unknown",
        "creator_kl_strategy": "none",
        "alpha_fair": 0.0,
        "delta_high": 0.0,
        "beta_prior_attention": 0.0,
        "tau_relevance": 1.0,
    }

    if not model_path.exists():
        defaults["run_time"] = infer_run_time_from_model_arg(model_arg)
        return defaults

    candidate_paths = [
        model_path / "run_config.json",
        model_path.parent / "run_config.json",
    ]
    cfg_path = next((path for path in candidate_paths if path.exists()), None)
    if cfg_path is None:
        return defaults

    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read run configuration: {cfg_path}") from exc
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid run configuration: {cfg_path}")
    if cfg.get("creator_kl_strategy") == "creator_anti_attention" and "beta_prior_attention" not in cfg:
        raise ValueError("Update the run configuration to the current prior attention schema before evaluation.")
    try:
        normalized = {**defaults, **{key: cfg[key] for key in defaults if key in cfg}}
        normalized["run_dir"] = cfg.get("run_dir", defaults["run_dir"])
        normalized["run_time"] = cfg.get(
            "run_time",
            infer_run_time_from_model_arg(str(cfg_path.parent / "model")),
        )
        normalized["adapter_mode"] = cfg.get(
            "adapter_mode",
            "qlora" if bool(cfg.get("use_qlora", 0)) else ("lora" if bool(cfg.get("use_lora", 0)) else "none"),
        )
        normalized["delta_high"] = cfg.get("delta_high", defaults["delta_high"])
        normalized["beta_prior_attention"] = cfg.get("beta_prior_attention", defaults["beta_prior_attention"])
        normalized["tau_relevance"] = cfg.get("tau_relevance", defaults["tau_relevance"])
        return normalized
    except Exception:
        defaults["run_time"] = infer_run_time_from_model_arg(model_arg)
        return defaults

