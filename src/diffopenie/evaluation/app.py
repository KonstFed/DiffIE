"""DiffIE demo: type a sentence, watch the reverse-diffusion trajectory, get triplets.

Usage:
    uv run streamlit run src/diffopenie/evaluation/app.py
    uv run streamlit run src/diffopenie/evaluation/app.py -- \
        --config configs/lsoie_ex_full/lsoie_ex_full_config.yaml \
        --checkpoint configs/lsoie_ex_full/weights.pt

Checkpoints come from ./scripts/fetch_artifacts.sh.
"""

from __future__ import annotations

import argparse
import html
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import torch

from diffopenie.evaluation.carb_eval import load_model
from diffopenie.models.discrete.extractors import LenientFrequencyExtractor
from diffopenie.training.train_example import TrainingConfig
from diffopenie.utils import load_config

DEFAULT_CONFIG = "configs/lsoie_ex_2500/lsoie_ex_2500_config.yaml"
DEFAULT_CHECKPOINT = "configs/lsoie_ex_2500/weights.pt"
EXAMPLE_SENTENCE = "Marie Curie discovered radium in 1898 ."

_STATE_ID_TO_STR = {0: "B", 1: "S", 2: "R", 3: "O", 4: "M"}
_TAG_COLORS: dict[str, tuple[str, str]] = {
    "B": ("#e0e0e0", "#444"),
    "S": ("#7ec8e3", "#003"),
    "R": ("#d4a8d4", "#300"),
    "O": ("#90d490", "#030"),
    "M": ("#666", "#ccc"),  # absorbing MASK state (MDLM)
    "P": ("#666", "#ccc"),  # PAD state (uniform)
}
_TAG_NAMES = {
    "B": "Background", "S": "Subject", "R": "Relation", "O": "Object",
    "M": "MASK", "P": "PAD",
}


def state_id_to_str(sid: int) -> str:
    return _STATE_ID_TO_STR.get(int(sid), "?")


# ── HTML helpers ───────────────────────────────────────────────────────────────


def _cell(tag: str) -> str:
    bg, fg = _TAG_COLORS.get(tag, ("#fff", "#000"))
    return (
        f'<td style="background:{bg};color:{fg};padding:2px 5px;'
        f"text-align:center;font-family:monospace;font-size:12px;"
        f'border:1px solid #ccc;white-space:nowrap">{html.escape(tag)}</td>'
    )


def _header_cell(text: str) -> str:
    return (
        f'<th style="padding:2px 5px;font-size:11px;font-family:monospace;'
        f'border:1px solid #ccc;white-space:nowrap;background:#f0f0f0">'
        f"{html.escape(text)}</th>"
    )


def _row_label_cell(text: str, bg: str = "#fafafa") -> str:
    return (
        f'<td style="padding:2px 6px;font-size:11px;font-family:monospace;'
        f"font-weight:bold;background:{bg};border:1px solid #ccc;"
        f'white-space:nowrap;position:sticky;left:0">{html.escape(text)}</td>'
    )


def build_diffusion_table(
    tokens: list[str],
    intermediates: torch.Tensor,  # [L, T]
    final_tags: list[str],
    uniform: bool,
) -> str:
    """One row per denoising step, from t=T (noise) down to the final tags."""
    rows = [
        "<tr>"
        + _row_label_cell("Step", bg="#e8e8e8")
        + "".join(_header_cell(tok) for tok in tokens)
        + "</tr>"
    ]

    def _tag(state: int) -> str:
        tag = state_id_to_str(state)
        return "P" if tag == "M" and uniform else tag

    num_steps = intermediates.shape[1]
    for step_idx in range(num_steps):
        tags = [_tag(int(s)) for s in intermediates[:, step_idx]]
        rows.append(
            "<tr>"
            + _row_label_cell(f"t={num_steps - step_idx}")
            + "".join(_cell(tag) for tag in tags)
            + "</tr>"
        )

    rows.append(
        "<tr>"
        + _row_label_cell("Final", bg="#e8f5e9")
        + "".join(_cell(tag) for tag in final_tags)
        + "</tr>"
    )
    return (
        "<div style='overflow-x:auto'>"
        "<table style='border-collapse:collapse;min-width:max-content'>"
        + "".join(rows)
        + "</table></div>"
    )


def legend_html(uniform: bool) -> str:
    tags = ["B", "S", "R", "O", "P" if uniform else "M"]
    return "".join(
        f'<span style="background:{_TAG_COLORS[t][0]};color:{_TAG_COLORS[t][1]};'
        f'padding:2px 8px;border-radius:3px;font-family:monospace;margin-right:6px">'
        f"{t} = {_TAG_NAMES[t]}</span>"
        for t in tags
    )


def _span_text(words: list[str], span: tuple | None) -> str:
    if span is None or span[0] is None:
        return "—"
    start, end = span
    return " ".join(words[start : end + 1])


def _tags_to_texts(tags: list[str], tokens: list[str]) -> tuple[str, str, str]:
    def extract(role: str) -> str:
        toks = [tokens[i] for i, t in enumerate(tags) if t == role]
        return " ".join(toks).replace(" ##", "") if toks else "—"

    return extract("S"), extract("R"), extract("O")


# ── Model ──────────────────────────────────────────────────────────────────────


@st.cache_resource
def load_cached_model(config_path: str, checkpoint_path: str):
    config = load_config(TrainingConfig, config_path)
    model = load_model(config, Path(checkpoint_path))
    model.eval()
    return model, config


def run_one_trajectory(model, token_ids: list[int], seed: int | None):
    """Single reverse-diffusion trajectory, keeping every intermediate step."""
    if seed is not None:
        torch.manual_seed(seed)
    ids = torch.tensor([token_ids], dtype=torch.long, device=model.device)
    attn = torch.ones_like(ids)
    with torch.no_grad():
        context = model.encode_tokens(ids, attn)
        x_0, intermediates = model.generate(
            batch_size=1,
            context=context,
            context_attention_mask=attn,
            tag_attention_mask=attn,
            return_intermediate=True,
        )
    return x_0[0].cpu(), intermediates[0].cpu()


# ── App ────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    # Streamlit injects its own argv; only parse the slice after "--"
    raw = sys.argv[1:]
    sep = raw.index("--") + 1 if "--" in raw else 0
    args, _ = parser.parse_known_args(raw[sep:])
    return args


def main() -> None:
    cli = parse_args()
    st.set_page_config(page_title="DiffIE Demo", layout="wide", page_icon="🔬")
    st.title("DiffIE — Open Information Extraction by discrete diffusion")

    with st.sidebar:
        st.header("Model")
        config_path = st.text_input("Config", cli.config)
        checkpoint_path = st.text_input("Checkpoint", cli.checkpoint)

        st.divider()
        st.header("Sampling")
        num_samples = st.select_slider(
            "Samples (n)", [1, 2, 4, 8, 16, 32, 64, 128, 256, 512], value=64,
            help="Independent reverse-diffusion trajectories clustered into the "
                 "final triplet set. More samples = better recall, slower.",
        )
        topk = st.slider("Triplets returned (k)", 1, 10, 4)
        threshold = st.slider("Lenient-match threshold (τ)", 0.5, 1.0, 0.9, 0.05)
        seed = st.number_input("Seed", min_value=0, value=42, step=1)
        use_seed = st.checkbox("Fixed seed", value=True)

    if not Path(checkpoint_path).exists():
        st.error(f"Checkpoint not found: `{checkpoint_path}`")
        st.info("Download the released checkpoints first:\n\n```bash\n"
                "./scripts/fetch_artifacts.sh\n```")
        return

    try:
        with st.spinner("Loading model..."):
            model, _ = load_cached_model(config_path, checkpoint_path)
    except Exception as exc:
        st.error(f"Failed to load model: {exc}")
        return

    sentence = st.text_area("Sentence", EXAMPLE_SENTENCE, height=80)
    # Runs on load and on every rerun (edit the sentence, move a slider, or
    # click Resample to draw a fresh set of trajectories).
    st.button("Resample", type="primary")
    if not sentence.strip():
        st.info("Enter a sentence above.")
        return

    words = sentence.split()
    tokenizer = model.encoder.tokenizer
    encoded = tokenizer(words, is_split_into_words=True, add_special_tokens=True)
    token_ids = encoded["input_ids"]
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    uniform = model.scheduler.kernel == "uniform"

    def fix(tag: str) -> str:
        return "P" if tag == "M" and uniform else tag

    with st.spinner("Denoising..."):
        x_0, intermediates = run_one_trajectory(
            model, token_ids, int(seed) if use_seed else None
        )
    final_tags = [fix(state_id_to_str(int(s))) for s in x_0]

    st.subheader("Reverse-diffusion trajectory")
    st.caption(
        f"One of the {num_samples} trajectories, from the noised tag sequence "
        f"(t={intermediates.shape[1]}) down to the final tags."
    )
    st.markdown(
        build_diffusion_table(tokens, intermediates, final_tags, uniform),
        unsafe_allow_html=True,
    )
    st.markdown(legend_html(uniform), unsafe_allow_html=True)

    subject, relation, obj = _tags_to_texts(final_tags, tokens)
    st.caption(
        f"This trajectory yields — **subject:** {subject} · "
        f"**relation:** {relation} · **object:** {obj}"
    )

    st.divider()
    st.subheader("Extracted triplets")
    st.caption(
        f"{num_samples} independent trajectories, clustered by lenient match "
        f"(τ={threshold}) and ranked by mass; top {topk} returned."
    )
    extractor = LenientFrequencyExtractor(k=int(num_samples), topk=int(topk),
                                          threshold=float(threshold))
    with st.spinner(f"Sampling {num_samples} trajectories..."):
        triplets, masses = extractor.get_carb_prediction(words, model.get_triplets)

    rows = [
        {
            "Score": f"{mass:.3f}",
            "Subject": _span_text(words, sub_span),
            "Relation": _span_text(words, pred_span),
            "Object": _span_text(words, obj_span),
        }
        for (sub_span, obj_span, pred_span), mass in zip(triplets, masses)
    ]
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    else:
        st.warning("No triplets extracted.")


if __name__ == "__main__":
    main()
