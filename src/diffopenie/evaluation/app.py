"""Interactive Streamlit debugger for DiffIE diffusion model.

Usage:
    uv run streamlit run src/diffopenie/evaluation/app.py
"""

from __future__ import annotations

import html
from pathlib import Path

import pandas as pd
import streamlit as st
import torch

from diffopenie.data.lsoie import SequenceLSOEIDataset, labels_to_indices
from diffopenie.evaluation.debug_diffusion import (
    load_carb_data,
    load_model,
    state_id_to_str,
)
from diffopenie.training.train_example import TrainingConfig
from diffopenie.utils import load_config

# ── Color scheme ───────────────────────────────────────────────────────────────

_TAG_COLORS: dict[str, tuple[str, str]] = {
    "B": ("#e0e0e0", "#444"),
    "S": ("#7ec8e3", "#003"),
    "R": ("#d4a8d4", "#300"),
    "O": ("#90d490", "#030"),
    "M": ("#666", "#ccc"),
}


def _cell(tag: str, text: str = "") -> str:
    bg, fg = _TAG_COLORS.get(tag, ("#fff", "#000"))
    label = html.escape(text or tag)
    return (
        f'<td style="background:{bg};color:{fg};padding:2px 5px;'
        f"text-align:center;font-family:monospace;font-size:12px;"
        f'border:1px solid #ccc;white-space:nowrap">{label}</td>'
    )


def _header_cell(text: str, sticky: bool = False) -> str:
    sticky_style = "position:sticky;top:0;" if sticky else ""
    return (
        f'<th style="{sticky_style}padding:2px 5px;font-size:11px;'
        f"font-family:monospace;border:1px solid #ccc;white-space:nowrap;"
        f'background:#f0f0f0">{html.escape(text)}</th>'
    )


def _row_label_cell(text: str, bg: str = "#fafafa") -> str:
    return (
        f'<td style="padding:2px 6px;font-size:11px;font-family:monospace;'
        f"font-weight:bold;background:{bg};border:1px solid #ccc;"
        f'white-space:nowrap;position:sticky;left:0">{html.escape(text)}</td>'
    )


def build_diffusion_table(
    tokens: list[str],
    gt_tags: list[str] | None,
    intermediates: torch.Tensor,  # [L, T]
    final_tags: list[str],
    T: int,
) -> str:
    rows: list[str] = []

    # Header: token names
    cells = _row_label_cell("Step", bg="#e8e8e8")
    for tok in tokens:
        cells += _header_cell(tok)
    rows.append(f"<tr>{cells}</tr>")

    # GT row
    if gt_tags is not None:
        cells = _row_label_cell("GT", bg="#fffde7")
        for tag in gt_tags:
            cells += _cell(tag)
        rows.append(f"<tr>{cells}</tr>")

    # Reverse diffusion steps
    for step_idx in range(T):
        ti = T - step_idx
        tags = [state_id_to_str(int(s)) for s in intermediates[:, step_idx]]
        cells = _row_label_cell(f"t={ti}", bg="#fafafa")
        for tag in tags:
            cells += _cell(tag)
        rows.append(f"<tr>{cells}</tr>")

    # Final row
    cells = _row_label_cell("Final", bg="#e8f5e9")
    for tag in final_tags:
        cells += _cell(tag)
    rows.append(f"<tr>{cells}</tr>")

    return (
        "<div style='overflow-x:auto'>"
        "<table style='border-collapse:collapse;min-width:max-content'>"
        + "".join(rows)
        + "</table></div>"
    )


# ── Helpers ────────────────────────────────────────────────────────────────────


def _span_text(words: list[str], span: tuple | None) -> str:
    if span is None or span[0] is None:
        return "—"
    s, e = span
    return " ".join(words[s : e + 1])


def _tags_to_token_texts(tags: list[str], tokens: list[str]) -> tuple[str, str, str]:
    def extract(role: str) -> str:
        toks = [tokens[i] for i, t in enumerate(tags) if t == role]
        return " ".join(toks).replace(" ##", "") if toks else "—"

    return extract("S"), extract("R"), extract("O")


def _legend_html() -> str:
    parts = []
    for tag, (bg, fg) in _TAG_COLORS.items():
        if tag == "M":
            label = "MASK"
        else:
            label = {"B": "Background", "S": "Subject", "R": "Relation", "O": "Object"}[
                tag
            ]
        parts.append(
            f'<span style="background:{bg};color:{fg};padding:2px 8px;'
            f'border-radius:3px;font-family:monospace;margin-right:6px">'
            f"{tag} = {label}</span>"
        )
    return "".join(parts)


# ── Model loading (cached) ─────────────────────────────────────────────────────


@st.cache_resource
def _load_model_cached(config_path: str, checkpoint_path: str):
    config = load_config(TrainingConfig, config_path)
    model = load_model(config, Path(checkpoint_path))
    model.eval()
    return model, config


# ── Generation ────────────────────────────────────────────────────────────────


def run_generation(
    model,
    token_ids: list[int],
    seed: int | None,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    if seed is not None:
        torch.manual_seed(seed)
    device = model.device
    ids_t = torch.tensor([token_ids], dtype=torch.long, device=device)
    attn = torch.ones_like(ids_t)
    with torch.no_grad():
        token_embeddings = model.encode_tokens(ids_t, attn)
        x_0, intermediates = model.generate(
            batch_size=1,
            token_embeddings=token_embeddings,
            attention_mask=attn,
            return_intermediate=True,
        )
    intermediates = intermediates[0].cpu()  # [L, T]
    x_0 = x_0[0].cpu()  # [L]
    final_tags = [state_id_to_str(int(s)) for s in x_0]
    return x_0, intermediates, final_tags


# ── App ───────────────────────────────────────────────────────────────────────


def main() -> None:
    st.set_page_config(page_title="DiffIE Debugger", layout="wide", page_icon="🔬")
    st.title("DiffIE — Interactive Diffusion Debugger")

    # ── Sidebar ────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Model")
        config_path = st.text_input("Config path", "config/training_mdlm.yaml")
        checkpoint_path = st.text_input(
            "Checkpoint path", "checkpoints/checkpoint_best.pt"
        )
        load_btn = st.button("Load / Reload Model", type="primary")

        st.divider()
        st.header("Input")
        dataset = st.selectbox("Dataset", ["lsoie", "carb", "custom"])

        split: str | None = None
        index: int = 0
        custom_sentence: str | None = None
        carb_dir_str: str | None = None

        if dataset in ("lsoie", "carb"):
            split = st.text_input("Split", "validation")
            index = int(st.number_input("Index", min_value=0, value=0, step=1))
        if dataset == "carb":
            carb_dir_str = st.text_input("CaRB dir (leave blank for auto-detect)", "")
        if dataset == "custom":
            custom_sentence = st.text_area(
                "Sentence", "The quick brown fox jumped over the lazy dog ."
            )

        st.divider()
        st.header("Options")
        use_seed = st.checkbox("Fixed seed", value=True)
        seed_val = int(st.number_input("Seed", min_value=0, value=42, step=1))
        actual_seed = seed_val if use_seed else None


    # ── Model loading ──────────────────────────────────────────────────────────
    if load_btn:
        _load_model_cached.clear()

    if not config_path or not checkpoint_path:
        st.info(
            "Enter config and checkpoint paths in the sidebar, then click **Load Model**."
        )
        return

    try:
        with st.spinner("Loading model..."):
            model, config = _load_model_cached(config_path, checkpoint_path)
    except Exception as exc:
        st.error(f"Failed to load model: {exc}")
        return

    tokenizer = model.encoder.tokenizer

    # ── Prepare example ────────────────────────────────────────────────────────
    try:
        tokens: list[str] = []
        token_ids: list[int] = []
        words: list[str] | None = None
        gt_labels: torch.Tensor | None = None
        all_gt_lsoie: list[list[str]] | None = None
        all_gt_carb: list[tuple[str, str, str]] | None = None

        if dataset == "custom":
            assert custom_sentence, "Enter a sentence."
            words = custom_sentence.split()
            enc = tokenizer(words, is_split_into_words=True, add_special_tokens=False)
            token_ids = enc["input_ids"]
            tokens = tokenizer.convert_ids_to_tokens(token_ids)

        elif dataset == "carb":
            carb_path = (
                Path(carb_dir_str)
                if carb_dir_str
                else Path(__file__).resolve().parents[3] / "CaRB"
            )
            sp = "dev" if split == "validation" else split
            sentences, gold_map = load_carb_data(
                carb_path / "data" / f"{sp}.txt",
                carb_path / "data" / "gold" / f"{sp}.tsv",
            )
            sentence = sentences[index]
            words = sentence.split()
            all_gt_carb = gold_map.get(sentence, []) or None
            enc = tokenizer(words, is_split_into_words=True, add_special_tokens=False)
            token_ids = enc["input_ids"]
            tokens = tokenizer.convert_ids_to_tokens(token_ids)

        else:  # lsoie
            ds = SequenceLSOEIDataset(
                split=split, tokenizer_name=config.model.encoder.model_name
            )
            item = ds[index]
            tokens = item["tokens"]
            token_ids = item["token_ids"]
            gt_labels = item["labels"]
            row = ds.dataset.iloc[index]
            words = row["words"]
            sentence = " ".join(words)
            matches = ds.dataset[ds.dataset["sentence"] == sentence]
            all_gt_lsoie = matches["label"].tolist()

    except Exception as exc:
        st.error(f"Failed to load example: {exc}")
        return

    # ── Generation ────────────────────────────────────────────────────────────
    # Re-generate when sentence changes or user clicks Reroll
    state_key = str(token_ids)
    reroll = st.button("🎲  Reroll generation", type="primary")

    need_gen = (
        reroll
        or "gen_key" not in st.session_state
        or st.session_state["gen_key"] != state_key
    )

    if need_gen:
        with st.spinner("Running diffusion..."):
            x_0, intermediates, final_tags = run_generation(
                model, token_ids, actual_seed
            )
        st.session_state.update(
            gen_key=state_key,
            x_0=x_0,
            intermediates=intermediates,
            final_tags=final_tags,
        )
        # Clear stale CaRB results when sentence changes
        if st.session_state.get("gen_key") != state_key:
            st.session_state.pop("carb_results", None)
    else:
        x_0 = st.session_state["x_0"]
        intermediates = st.session_state["intermediates"]
        final_tags = st.session_state["final_tags"]

    T = model.scheduler.num_steps
    gt_tags: list[str] | None = (
        [state_id_to_str(int(s)) for s in gt_labels] if gt_labels is not None else None
    )

    # ── Tabs ───────────────────────────────────────────────────────────────────
    tab_diff, tab_triplets, tab_carb = st.tabs(
        ["🎞  Diffusion Steps", "📋  Triplets", "📊  CaRB Prediction"]
    )

    # ── Tab 1: Diffusion grid ──────────────────────────────────────────────────
    with tab_diff:
        if words:
            st.markdown(f"**Sentence:** {' '.join(words)}")

        if gt_labels is not None:
            correct = (x_0 == gt_labels).sum().item()
            total = len(gt_labels)
            st.metric(
                "Token accuracy", f"{100 * correct / total:.1f}%", f"{correct}/{total}"
            )

        st.markdown(
            build_diffusion_table(tokens, gt_tags, intermediates, final_tags, T),
            unsafe_allow_html=True,
        )
        st.markdown(_legend_html(), unsafe_allow_html=True)

        # Per-class breakdown
        if gt_labels is not None:
            with st.expander("Per-class accuracy"):
                rows_acc = []
                for name, tid in [("B", 0), ("S", 1), ("R", 2), ("O", 3)]:
                    mask = gt_labels == tid
                    if mask.sum() == 0:
                        continue
                    n_correct = int(((x_0 == tid) & mask).sum())
                    n_total = int(mask.sum())
                    rows_acc.append(
                        {
                            "Tag": name,
                            "Correct": n_correct,
                            "Total": n_total,
                            "Acc": f"{100 * n_correct / n_total:.1f}%",
                        }
                    )
                st.dataframe(pd.DataFrame(rows_acc), hide_index=True)

    # ── Tab 2: Triplets ────────────────────────────────────────────────────────
    with tab_triplets:
        if words is None:
            st.info("Triplet extraction requires a sentence with word boundaries.")
        else:
            col_pred, col_gt = st.columns(2)

            with col_pred:
                st.subheader("Predicted (this roll)")
                pred_s, pred_r, pred_o = _tags_to_token_texts(final_tags, tokens)
                st.markdown(
                    f"**Subject:** {pred_s}  \n**Relation:** {pred_r}  \n**Object:** {pred_o}"
                )

            with col_gt:
                st.subheader("Ground Truth")
                if all_gt_lsoie:
                    rows_gt = []
                    for labels in all_gt_lsoie:
                        sub_span, obj_span, pred_span = labels_to_indices(labels)
                        rows_gt.append(
                            {
                                "Subject": _span_text(words, sub_span),
                                "Relation": _span_text(words, pred_span),
                                "Object": _span_text(words, obj_span),
                            }
                        )
                    st.dataframe(
                        pd.DataFrame(rows_gt), hide_index=True, use_container_width=True
                    )
                elif all_gt_carb:
                    rows_gt = [
                        {"Subject": s, "Relation": p, "Object": o}
                        for s, p, o in all_gt_carb
                    ]
                    st.dataframe(
                        pd.DataFrame(rows_gt), hide_index=True, use_container_width=True
                    )
                else:
                    st.info("No ground truth available.")

    # ── Tab 3: CaRB prediction ─────────────────────────────────────────────────
    with tab_carb:
        if words is None:
            st.info("CaRB prediction requires a sentence.")
        else:
            st.caption(
                "Samples model multiple times and returns the most frequent triplets."
            )
            run_carb = st.button("▶  Run CaRB prediction")

            if run_carb:
                with st.spinner("Sampling..."):
                    triplets, probs = model.get_carb_prediction(words)
                st.session_state["carb_results"] = (triplets, probs, list(words))

            if "carb_results" in st.session_state:
                cached_triplets, cached_probs, cached_words = st.session_state[
                    "carb_results"
                ]

                pred_rows = []
                for (sub_span, obj_span, pred_span), prob in zip(
                    cached_triplets, cached_probs
                ):
                    pred_rows.append(
                        {
                            "Prob": f"{prob:.3f}",
                            "Subject": _span_text(cached_words, sub_span),
                            "Relation": _span_text(cached_words, pred_span),
                            "Object": _span_text(cached_words, obj_span),
                        }
                    )

                if pred_rows:
                    st.subheader("Extracted triplets")
                    st.dataframe(
                        pd.DataFrame(pred_rows),
                        hide_index=True,
                        use_container_width=True,
                    )
                else:
                    st.warning("No triplets extracted.")

                # Gold comparison
                if all_gt_carb or all_gt_lsoie:
                    st.subheader("Gold triplets")
                    if all_gt_carb:
                        gold_rows = [
                            {"Subject": s, "Relation": p, "Object": o}
                            for s, p, o in all_gt_carb
                        ]
                    else:
                        gold_rows = []
                        for labels in all_gt_lsoie:  # type: ignore[union-attr]
                            sub_span, obj_span, pred_span = labels_to_indices(labels)
                            gold_rows.append(
                                {
                                    "Subject": _span_text(cached_words, sub_span),
                                    "Relation": _span_text(cached_words, pred_span),
                                    "Object": _span_text(cached_words, obj_span),
                                }
                            )
                    st.dataframe(
                        pd.DataFrame(gold_rows),
                        hide_index=True,
                        use_container_width=True,
                    )


if __name__ == "__main__":
    main()
