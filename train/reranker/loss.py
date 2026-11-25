from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F


IGNORE_INDEX = -100
VALID_LOSS_TYPES = {"ntp", "ntp_kl"}
VALID_CREATOR_KL_STRATEGIES = {"none", "creator_anti_attention"}
VALID_KL_TARGET_STRATEGIES = {"fair_group", "high_suppressed_group"}
NUM_VISIBILITY_GROUPS = 3


def validate_loss_type(loss_type: str) -> str:
    if loss_type not in VALID_LOSS_TYPES:
        supported = ", ".join(sorted(VALID_LOSS_TYPES))
        raise ValueError(f"Unsupported loss_type '{loss_type}'. Supported values: {supported}")
    return loss_type


def validate_kl_target_strategy(kl_target_strategy: str) -> str:
    if kl_target_strategy not in VALID_KL_TARGET_STRATEGIES:
        supported = ", ".join(sorted(VALID_KL_TARGET_STRATEGIES))
        raise ValueError(
            "Unsupported kl_target_strategy "
            f"'{kl_target_strategy}'. Supported values: {supported}"
        )
    return kl_target_strategy


def validate_creator_kl_strategy(creator_kl_strategy: str) -> str:
    if creator_kl_strategy not in VALID_CREATOR_KL_STRATEGIES:
        supported = ", ".join(sorted(VALID_CREATOR_KL_STRATEGIES))
        raise ValueError(
            "Unsupported creator_kl_strategy "
            f"'{creator_kl_strategy}'. Supported values: {supported}"
        )
    return creator_kl_strategy


def loss_uses_kl_target_strategy(loss_type: str) -> bool:
    return "kl" in loss_type


def normalize_kl_target_strategy(loss_type: str, kl_target_strategy: str | None) -> str:
    if not loss_uses_kl_target_strategy(loss_type):
        return "none"
    return validate_kl_target_strategy(str(kl_target_strategy or "high_suppressed_group"))


def normalize_creator_kl_strategy(creator_kl_strategy: str | None) -> str:
    return validate_creator_kl_strategy(str(creator_kl_strategy or "none"))


def _get_shifted_logits_and_labels(logits, labels):
    return logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()


def get_creator_step_positions(labels, ignore_index=IGNORE_INDEX):
    shift_labels = labels[..., 1:].contiguous()
    valid_positions = shift_labels.ne(ignore_index)
    has_valid_target = valid_positions.any(dim=1)
    creator_step_positions = [
        torch.nonzero(sample_valid_positions, as_tuple=False).squeeze(-1)
        for sample_valid_positions in valid_positions
    ]
    return shift_labels, creator_step_positions, has_valid_target


def compute_restricted_ntp_loss(
    logits,
    labels,
    allowed_token_ids,
    ignore_index=IGNORE_INDEX,
):
    """Compute creator-generation NTP over all dataset creator tokens plus EOS only."""
    shift_logits, shift_labels = _get_shifted_logits_and_labels(logits, labels)
    valid_positions = shift_labels.ne(ignore_index)
    if not valid_positions.any():
        return logits.new_zeros(())

    allowed_token_ids = allowed_token_ids.to(logits.device)
    restricted_logits = shift_logits.index_select(dim=-1, index=allowed_token_ids)

    vocab_size = logits.size(-1)
    full_to_restricted = torch.full(
        (vocab_size,),
        fill_value=-1,
        dtype=torch.long,
        device=logits.device,
    )
    full_to_restricted[allowed_token_ids] = torch.arange(
        allowed_token_ids.numel(),
        device=logits.device,
    )

    safe_labels = shift_labels.masked_fill(~valid_positions, 0)
    restricted_labels = full_to_restricted[safe_labels]
    invalid_targets = valid_positions & restricted_labels.lt(0)
    if invalid_targets.any():
        bad_ids = torch.unique(shift_labels[invalid_targets]).tolist()
        raise ValueError(
            "Restricted NTP received labels outside creator-token space. "
            f"Unexpected token ids: {bad_ids}"
        )

    flat_logits = restricted_logits[valid_positions]
    flat_labels = restricted_labels[valid_positions]
    return F.cross_entropy(flat_logits, flat_labels)


def _build_group_memberships(creator_scores, group_ids):
    groups = torch.as_tensor(group_ids, device=creator_scores.device)
    if groups.ndim == 1:
        if groups.numel() != creator_scores.numel():
            raise ValueError("group ids must align with project creators.")
        groups = groups.to(dtype=torch.long)
        return F.one_hot(groups, num_classes=NUM_VISIBILITY_GROUPS).to(dtype=creator_scores.dtype)
    if groups.ndim == 2:
        if groups.shape != (creator_scores.numel(), NUM_VISIBILITY_GROUPS):
            raise ValueError(
                "Soft visibility-group memberships must have shape "
                f"[num_creators, {NUM_VISIBILITY_GROUPS}]."
            )
        memberships = groups.to(dtype=creator_scores.dtype)
        row_sums = memberships.sum(dim=-1, keepdim=True)
        if torch.any(row_sums <= 0):
            raise ValueError("Soft visibility-group memberships must have positive row sums.")
        return memberships / row_sums
    raise ValueError("group_ids must be either hard group ids or per-group memberships.")


def _compute_relevance_group_mass(creator_scores, relevance_labels, group_ids):
    relevance = torch.as_tensor(
        relevance_labels,
        dtype=creator_scores.dtype,
        device=creator_scores.device,
    )
    if creator_scores.numel() != relevance.numel():
        raise ValueError("relevance labels and group ids must align with project creators.")

    memberships = _build_group_memberships(creator_scores, group_ids)
    total_relevance_mass = relevance.sum()
    if total_relevance_mass <= 0:
        return None, memberships

    epsilon_rel = (relevance.unsqueeze(-1) * memberships).sum(dim=0) / total_relevance_mass
    return epsilon_rel, memberships


def _compute_fair_group_target(epsilon_rel, alpha_fair):
    uniform = torch.full(
        (NUM_VISIBILITY_GROUPS,),
        fill_value=1.0 / NUM_VISIBILITY_GROUPS,
        dtype=epsilon_rel.dtype,
        device=epsilon_rel.device,
    )
    return (1.0 - float(alpha_fair)) * epsilon_rel + float(alpha_fair) * uniform


def _compute_high_suppressed_group_target(epsilon_rel, delta_high):
    target = epsilon_rel.clone()
    delta = torch.as_tensor(
        float(delta_high),
        dtype=epsilon_rel.dtype,
        device=epsilon_rel.device,
    )
    removed_mass = torch.minimum(target[2], delta)
    target[2] = target[2] - removed_mass
    target[0] = target[0] + removed_mass / 2.0
    target[1] = target[1] + removed_mass / 2.0
    target_sum = target.sum()
    if target_sum <= 0:
        return None
    return target / target_sum


def _compute_group_target_distribution(
    creator_scores,
    relevance_labels,
    group_ids,
    kl_target_strategy,
    alpha_fair,
    delta_high,
):
    strategy = validate_kl_target_strategy(kl_target_strategy)
    epsilon_rel, memberships = _compute_relevance_group_mass(
        creator_scores=creator_scores,
        relevance_labels=relevance_labels,
        group_ids=group_ids,
    )
    if epsilon_rel is None:
        return None, memberships

    if strategy == "fair_group":
        return _compute_fair_group_target(epsilon_rel, alpha_fair), memberships
    return _compute_high_suppressed_group_target(epsilon_rel, delta_high), memberships


def _compute_group_kl_from_scores(
    creator_scores,
    relevance_labels,
    group_ids,
    kl_target_strategy,
    alpha_fair,
    delta_high,
    eps=1e-8,
):
    epsilon_target, memberships = _compute_group_target_distribution(
        creator_scores=creator_scores,
        relevance_labels=relevance_labels,
        group_ids=group_ids,
        kl_target_strategy=kl_target_strategy,
        alpha_fair=alpha_fair,
        delta_high=delta_high,
    )
    if epsilon_target is None:
        return None

    creator_probs = F.softmax(creator_scores, dim=-1)
    epsilon_pred = (creator_probs.unsqueeze(-1) * memberships).sum(dim=0)
    valid_groups = epsilon_target.gt(0)
    if not valid_groups.any():
        return None

    safe_target = epsilon_target[valid_groups].clamp_min(eps)
    safe_pred = epsilon_pred[valid_groups].clamp_min(eps)
    return torch.sum(safe_target * (torch.log(safe_target) - torch.log(safe_pred)))


def _compute_creator_anti_attention_kl_from_scores(
    creator_scores,
    relevance_labels,
    prior_attention_scores,
    beta_prior_attention,
    tau_relevance,
    eps=1e-8,
):
    """Project-wide KL(q || p), with q_i proportional to relevance * exp(-attention)."""
    relevance = torch.as_tensor(
        relevance_labels, dtype=creator_scores.dtype, device=creator_scores.device
    )
    attention = torch.as_tensor(
        prior_attention_scores, dtype=creator_scores.dtype, device=creator_scores.device
    )
    if creator_scores.numel() != relevance.numel() or creator_scores.numel() != attention.numel():
        raise ValueError("Relevance and attention values must align with project creators.")
    if torch.any(relevance < 0):
        raise ValueError("Creator relevance labels must be non-negative.")
    if relevance.sum() <= 0:
        return None

    # The model works with project-normalized historical attention.
    min_attention, max_attention = attention.min(), attention.max()
    attention = (attention - min_attention) / (max_attention - min_attention).clamp_min(eps)
    relevance_weight = relevance if float(tau_relevance) == 1.0 else relevance.pow(float(tau_relevance))
    target_weights = relevance_weight * torch.exp(-float(beta_prior_attention) * attention)
    total_target_weight = target_weights.sum()
    if total_target_weight <= 0:
        return None

    target_probs = target_weights / total_target_weight
    log_pred_probs = F.log_softmax(creator_scores, dim=-1)
    nonzero = target_probs > 0
    return torch.sum(
        target_probs[nonzero] * (target_probs[nonzero].log() - log_pred_probs[nonzero])
    )


def _iter_first_step_project_candidates(
    logits,
    labels,
    project_creator_token_ids,
    aligned_values,
    ignore_index=IGNORE_INDEX,
):
    """Yield one project-local creator distribution at the first labeled creator token."""
    shift_logits, shift_labels = _get_shifted_logits_and_labels(logits, labels)
    _, positions_per_sample, has_valid_target = get_creator_step_positions(
        labels, ignore_index=ignore_index
    )

    if len(project_creator_token_ids) != len(aligned_values):
        raise ValueError("Project creator IDs and their aligned values must match in batch size.")

    for batch_idx, token_ids in enumerate(project_creator_token_ids):
        if not bool(has_valid_target[batch_idx]) or not token_ids:
            continue
        if len(token_ids) != len(aligned_values[batch_idx]):
            raise ValueError("Aligned project values must match project creator token IDs.")
        first_pos = int(positions_per_sample[batch_idx][0])
        creator_ids = [int(token_id) for token_id in token_ids]
        target_token_id = int(shift_labels[batch_idx, first_pos])
        if target_token_id not in creator_ids:
            # For example, a sample with no creator labels has EOS as its first target.
            continue

        creator_scores = shift_logits[batch_idx, first_pos].index_select(
            0, torch.tensor(creator_ids, dtype=torch.long, device=logits.device)
        )
        yield creator_scores, aligned_values[batch_idx]


def compute_project_kl_loss(
    logits,
    labels,
    project_creator_token_ids,
    relevance_labels,
    group_ids,
    kl_target_strategy,
    alpha_fair,
    delta_high,
    ignore_index=IGNORE_INDEX,
):
    """Mean first-step group KL over the batch, restricted to project creators."""
    aligned_values = [
        list(zip(sample_relevance, sample_group_ids))
        for sample_relevance, sample_group_ids in zip(relevance_labels, group_ids)
    ]
    losses = []
    for scores, values in _iter_first_step_project_candidates(
        logits, labels, project_creator_token_ids, aligned_values, ignore_index
    ):
        step_relevance, step_groups = zip(*values)
        kl = _compute_group_kl_from_scores(
            creator_scores=scores,
            relevance_labels=step_relevance,
            group_ids=step_groups,
            kl_target_strategy=kl_target_strategy,
            alpha_fair=alpha_fair,
            delta_high=delta_high,
        )
        if kl is not None:
            losses.append(kl)
    return torch.stack(losses).mean() if losses else logits.new_zeros(())


def compute_project_creator_kl_loss(
    logits,
    labels,
    project_creator_token_ids,
    relevance_labels,
    prior_attention_scores,
    creator_kl_strategy,
    beta_prior_attention,
    tau_relevance,
    ignore_index=IGNORE_INDEX,
):
    """Mean project-wide creator KL using the first creator generation step only."""
    if normalize_creator_kl_strategy(creator_kl_strategy) == "none":
        return logits.new_zeros(())

    aligned_values = [
        list(zip(sample_relevance, sample_attention))
        for sample_relevance, sample_attention in zip(relevance_labels, prior_attention_scores)
    ]
    losses = []
    for scores, values in _iter_first_step_project_candidates(
        logits, labels, project_creator_token_ids, aligned_values, ignore_index
    ):
        step_relevance, step_attention = zip(*values)
        kl = _compute_creator_anti_attention_kl_from_scores(
            creator_scores=scores,
            relevance_labels=step_relevance,
            prior_attention_scores=step_attention,
            beta_prior_attention=beta_prior_attention,
            tau_relevance=tau_relevance,
        )
        if kl is not None:
            losses.append(kl)
    return torch.stack(losses).mean() if losses else logits.new_zeros(())


def compute_total_loss(
    *,
    logits,
    labels,
    all_creator_token_ids,
    project_creator_token_ids,
    rank_labels,
    project_prior_attention_scores,
    project_group_ids,
    loss_type="ntp",
    lambda_ntp=1.0,
    lambda_kl=1.0,
    lambda_kl_creator=0.0,
    kl_target_strategy="none",
    creator_kl_strategy="none",
    alpha_fair=0.2,
    delta_high=0.05,
    beta_prior_attention=1.0,
    tau_relevance=1.0,
    eos_token_id=None,
    ignore_index=IGNORE_INDEX,
):
    validate_loss_type(loss_type)
    kl_target_strategy = normalize_kl_target_strategy(loss_type, kl_target_strategy)
    creator_kl_strategy = normalize_creator_kl_strategy(creator_kl_strategy)

    if eos_token_id is not None:
        all_creator_token_id_set = {int(token_id) for token_id in all_creator_token_ids.tolist()}
        if int(eos_token_id) not in all_creator_token_id_set:
            raise ValueError("all_creator_token_ids must include EOS.")
        for batch_idx, token_ids in enumerate(project_creator_token_ids):
            if int(eos_token_id) in {int(token_id) for token_id in token_ids}:
                raise ValueError(
                    "project_creator_token_ids must not include EOS. "
                    f"Found EOS in batch index {batch_idx}."
                )

    ntp_loss = compute_restricted_ntp_loss(
        logits=logits,
        labels=labels,
        allowed_token_ids=all_creator_token_ids,
        ignore_index=ignore_index,
    )

    zero = logits.new_zeros(())
    kl_loss = zero
    creator_kl_loss = zero
    total_loss = lambda_ntp * ntp_loss

    if loss_type == "ntp_kl":
        kl_loss = compute_project_kl_loss(
            logits=logits,
            labels=labels,
            project_creator_token_ids=project_creator_token_ids,
            relevance_labels=rank_labels,
            group_ids=project_group_ids,
            kl_target_strategy=kl_target_strategy,
            alpha_fair=alpha_fair,
            delta_high=delta_high,
            ignore_index=ignore_index,
        )
        total_loss = total_loss + lambda_kl * kl_loss

        creator_kl_active = (
            float(lambda_kl_creator) != 0.0
            and creator_kl_strategy == "creator_anti_attention"
        )
        if creator_kl_active:
            creator_kl_loss = compute_project_creator_kl_loss(
                logits=logits,
                labels=labels,
                project_creator_token_ids=project_creator_token_ids,
                relevance_labels=rank_labels,
                prior_attention_scores=project_prior_attention_scores,
                creator_kl_strategy=creator_kl_strategy,
                beta_prior_attention=beta_prior_attention,
                tau_relevance=tau_relevance,
                ignore_index=ignore_index,
            )
            total_loss = total_loss + lambda_kl_creator * creator_kl_loss

    return total_loss, {
        "loss": total_loss.detach(),
        "ntp_loss": ntp_loss.detach(),
        "kl_loss": kl_loss.detach(),
        "creator_kl_loss": creator_kl_loss.detach(),
    }


class CreatorRankingLoss:
    def __init__(
        self,
        mode="ntp",
        lambda_ntp=1.0,
        lambda_kl=1.0,
        lambda_kl_creator=0.0,
        kl_target_strategy="none",
        creator_kl_strategy="none",
        alpha_fair=0.2,
        delta_high=0.05,
        beta_prior_attention=1.0,
        tau_relevance=1.0,
        eos_token_id=None,
        ignore_index=IGNORE_INDEX,
    ):
        self.mode = validate_loss_type(mode)
        self.lambda_ntp = lambda_ntp
        self.lambda_kl = lambda_kl
        self.lambda_kl_creator = float(lambda_kl_creator)
        self.kl_target_strategy = normalize_kl_target_strategy(self.mode, kl_target_strategy)
        self.creator_kl_strategy = normalize_creator_kl_strategy(creator_kl_strategy)
        self.alpha_fair = float(alpha_fair)
        self.delta_high = float(delta_high)
        self.beta_prior_attention = float(beta_prior_attention)
        self.tau_relevance = float(tau_relevance)
        self.eos_token_id = eos_token_id
        self.ignore_index = ignore_index

    def __call__(
        self,
        *,
        logits,
        labels,
        all_creator_token_ids,
        project_creator_token_ids,
        rank_labels,
        project_prior_attention_scores,
        project_group_ids,
    ):
        total_loss, loss_dict = compute_total_loss(
            logits=logits,
            labels=labels,
            all_creator_token_ids=all_creator_token_ids,
            project_creator_token_ids=project_creator_token_ids,
            rank_labels=rank_labels,
            project_prior_attention_scores=project_prior_attention_scores,
            project_group_ids=project_group_ids,
            loss_type=self.mode,
            lambda_ntp=self.lambda_ntp,
            lambda_kl=self.lambda_kl,
            lambda_kl_creator=self.lambda_kl_creator,
            kl_target_strategy=self.kl_target_strategy,
            creator_kl_strategy=self.creator_kl_strategy,
            alpha_fair=self.alpha_fair,
            delta_high=self.delta_high,
            beta_prior_attention=self.beta_prior_attention,
            tau_relevance=self.tau_relevance,
            eos_token_id=self.eos_token_id,
            ignore_index=self.ignore_index,
        )
        out = {
            "loss": total_loss,
            "ntp_loss": loss_dict["ntp_loss"],
            "kl_loss": loss_dict["kl_loss"],
        }
        if (
            self.lambda_kl_creator != 0.0
            and self.creator_kl_strategy == "creator_anti_attention"
        ):
            out["creator_kl_loss"] = loss_dict["creator_kl_loss"]
        return out
