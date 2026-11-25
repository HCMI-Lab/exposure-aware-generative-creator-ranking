"""Prepare prior attention inputs without relying on dataset directory names."""
from pathlib import Path
import logging

import pandas as pd

log = logging.getLogger(__name__)


def preprocess(data_root: str, attention_source: str = "auto") -> None:
    """Normalize supported raw schemas or use an existing prior attention file.

    The source is selected explicitly or inferred from the input columns. An
    existing file is preferred in auto mode to avoid changing supplied scores.
    """
    data_root = Path(data_root)
    details_path = data_root / "creator_details.csv"
    attention_path = data_root / "prior_attention_scores.csv"
    df = pd.read_csv(details_path)
    if attention_source == "auto":
        if attention_path.exists():
            attention_source = "provided"
        else:
            sources = [source for source, column in (
                ("total_engagement", "total_engagement"), ("mindshare", "Mindshare")
            ) if column in df.columns]
            if len(sources) != 1:
                raise ValueError("Supply prior_attention_scores.csv or explicitly select an attention source.")
            attention_source = sources[0]

    if attention_source not in {"provided", "total_engagement", "mindshare"}:
        raise ValueError(f"Unsupported attention source: {attention_source}")

    changed = False
    if "Creator_ID" not in df.columns:
        if attention_source == "total_engagement":
            handle_col = "twitter_handle"
        elif attention_source == "mindshare":
            handle_col = "Handle"
        else:
            handles = [col for col in ("twitter_handle", "Handle") if col in df.columns]
            if len(handles) != 1:
                raise ValueError("Creator_ID is required when the handle column is ambiguous or missing.")
            handle_col = handles[0]
        if handle_col not in df.columns:
            raise ValueError(f"Missing creator identifier column: {handle_col}")
        df["Creator_ID"] = df[handle_col].astype(str).str.strip().str.lower().str.lstrip("@")
        changed = True

    if attention_source == "mindshare":
        required = {"project_slug", "Mindshare"}
        if not required.issubset(df.columns):
            raise ValueError(f"Missing columns: {sorted(required - set(df.columns))}")
        if df["Mindshare"].dtype == object:
            df["Mindshare"] = df["Mindshare"].astype(str).str.replace("%", "", regex=False).astype(float)
        df["Rank"] = (
            df.sort_values(by=["project_slug", "Mindshare"], ascending=[True, False])
            .groupby("project_slug").cumcount() + 1
        )
        changed = True

    if attention_source == "provided":
        if not attention_path.exists():
            raise FileNotFoundError(f"Missing prior attention file: {attention_path}")
    elif not attention_path.exists():
        source_col = "total_engagement" if attention_source == "total_engagement" else "Mindshare"
        if source_col not in df.columns:
            raise ValueError(f"Missing prior attention source column: {source_col}")
        attention_df = (
            df.groupby("Creator_ID", as_index=False)[source_col].mean()
            .rename(columns={"Creator_ID": "username", source_col: "prior_attention_score"})
        )
        attention_df["username"] = attention_df["username"].astype(str).str.lower().str.lstrip("@")
        attention_df["prior_attention_score"] = pd.to_numeric(
            attention_df["prior_attention_score"], errors="coerce"
        ).fillna(0.0)
        attention_df.sort_values("prior_attention_score", ascending=False).to_csv(attention_path, index=False)
        log.info("Generated %s with %s creators", attention_path, len(attention_df))

    if changed:
        df.to_csv(details_path, index=False)
