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
import random
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st
import torch

from diffopenie.data import SEQ_STR2INT
from diffopenie.data.triplet_utils import extract_longest_span
from diffopenie.evaluation.carb_eval import load_model
from diffopenie.models.discrete.extractors import LenientFrequencyExtractor
from diffopenie.training.train_example import TrainingConfig
from diffopenie.utils import load_config

DEFAULT_CONFIG = "configs/lsoie_ex_2500/lsoie_ex_2500_config.yaml"
DEFAULT_CHECKPOINT = "configs/lsoie_ex_2500/weights.pt"
EXAMPLE_SENTENCE = "Marie Curie discovered radium in 1898 ."

_STATE_ID_TO_STR = {0: "B", 1: "S", 2: "R", 3: "O", 4: "M"}
_TAG_NAMES = {
    "B": "Background", "S": "Subject", "R": "Relation", "O": "Object",
    "M": "MASK", "P": "PAD",
}
# Categorical slots 1-3 of the validated reference palette, per theme. Roles S/R/O
# are the only real categories; B and M/P are "no role yet" and stay neutral.
# Validated all-pairs in both modes (worst CVD dE 9.2 light / 9.4 dark).
_PALETTES: dict[str, dict[str, tuple[str, str]]] = {
    "light": {
        "B": ("#d9d9d6", "#3a3a38"),
        "S": ("#2a78d6", "#111111"),
        "R": ("#eb6834", "#111111"),
        "O": ("#1baf7a", "#111111"),
        "M": ("#9a9a96", "#1c1c1a"),
        "P": ("#9a9a96", "#1c1c1a"),
    },
    "dark": {
        "B": ("#3f3f3c", "#e8e8e4"),
        "S": ("#3987e5", "#111111"),
        "R": ("#d95926", "#111111"),
        "O": ("#199e70", "#111111"),
        "M": ("#6d6d69", "#f0f0ec"),
        "P": ("#6d6d69", "#f0f0ec"),
    },
}


def state_id_to_str(sid: int) -> str:
    return _STATE_ID_TO_STR.get(int(sid), "?")


def palette() -> dict[str, tuple[str, str]]:
    """Theme-selected role colors (each mode's own validated steps, not a flip)."""
    theme = getattr(getattr(st, "context", None), "theme", None)
    mode = getattr(theme, "type", None) or "light"
    return _PALETTES.get(mode, _PALETTES["light"])


# ── HTML pieces ────────────────────────────────────────────────────────────────


def chip(text: str, tag: str, pal, *, size: str = "13px") -> str:
    bg, fg = pal.get(tag, ("#d9d9d6", "#111"))
    dotted = ";border:1px dashed rgba(0,0,0,.35)" if tag in ("M", "P") else ""
    return (
        f'<span style="background:{bg};color:{fg};padding:3px 9px;border-radius:6px;'
        f'font-size:{size};font-weight:600;display:inline-block;margin:2px 3px 2px 0'
        f'{dotted}">{html.escape(text)}</span>'
    )


def render_matrix(
    headers: list[str],
    rows: list[tuple[str, list[str]]],
    pal,
    *,
    col_w: int = 52,
) -> str:
    """A uniform grid: one equal-sized block per (step, token).

    Every block is the same size regardless of how long its word is, so the
    trajectory reads as a picture — noise at the top resolving into roles at the
    bottom — instead of a ragged line of text.
    """
    def header(text: str) -> str:
        return (
            f'<th title="{html.escape(text)}" style="width:{col_w}px;'
            f"max-width:{col_w}px;padding:0 2px 6px;"
            f"font:600 11px/1.2 -apple-system,system-ui,sans-serif;"
            f"opacity:.75;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;"
            f'text-align:center">{html.escape(text)}</th>'
        )

    def block(tag: str) -> str:
        bg, fg = pal.get(tag, ("#d9d9d6", "#111"))
        dashed = ";outline:1px dashed rgba(128,128,128,.45);outline-offset:-1px" \
            if tag in ("M", "P") else ""
        return (
            f'<td style="padding:0"><div style="height:20px;border-radius:4px;'
            f"background:{bg};color:{fg};font:700 10px/20px monospace;"
            f'text-align:center{dashed}">{tag}</div></td>'
        )

    def row_label(text: str, strong: bool = False) -> str:
        weight = "700" if strong else "500"
        return (
            f'<td style="padding:0 8px 0 0;font:{weight} 10px/20px monospace;'
            f'opacity:.55;text-align:right;white-space:nowrap">{html.escape(text)}</td>'
        )

    out = ["<tr><td></td>" + "".join(header(h) for h in headers) + "</tr>"]
    for i, (label, tags) in enumerate(rows):
        strong = i == len(rows) - 1
        out.append(
            "<tr>" + row_label(label, strong)
            + "".join(block(t) for t in tags) + "</tr>"
        )
    return (
        "<div style='overflow-x:auto;padding:4px 0'>"
        "<table style='border-spacing:3px;border-collapse:separate;"
        "table-layout:fixed;min-width:max-content'>"
        + "".join(out) + "</table></div>"
    )


def render_legend(pal, tags: list[str]) -> str:
    return (
        "<div style='margin-top:6px'>"
        + "".join(chip(f"{t} · {_TAG_NAMES[t]}", t, pal, size="11px") for t in tags)
        + "</div>"
    )


def render_triplet_card(mass: float, subject: str, relation: str, obj: str, pal) -> str:
    pct = max(2.0, min(100.0, mass * 100))
    bar, _ = pal["S"]
    track, _ = pal["B"]
    return (
        "<div>"
        "<div style='display:flex;align-items:center;gap:10px'>"
        f"<div style='flex:0 0 160px;height:8px;border-radius:4px;background:{track}'>"
        f"<div style='width:{pct:.1f}%;height:8px;border-radius:4px;"
        f"background:{bar}'></div></div>"
        f"<span style='font:13px monospace;opacity:.75'>{mass:.3f}</span>"
        "</div>"
        "<div style='margin-top:8px'>"
        + chip(subject, "S", pal) + chip(relation, "R", pal) + chip(obj, "O", pal)
        + "</div></div>"
    )


def fix_tag(tag: str, uniform: bool) -> str:
    return "P" if tag == "M" and uniform else tag


def span_text(words: list[str], span: tuple | None) -> str:
    if span is None or span[0] is None:
        return "—"
    start, end = span
    return " ".join(words[start : end + 1])


# ── Model & sampling ───────────────────────────────────────────────────────────


@st.cache_resource(show_spinner=False)
def load_cached_model(config_path: str, checkpoint_path: str):
    config = load_config(TrainingConfig, config_path)
    model = load_model(config, Path(checkpoint_path))
    model.eval()
    return model, config


@st.cache_data(show_spinner=False)
def draw_trajectories(config_path: str, checkpoint_path: str, sentence: str,
                      n: int, seed: int) -> dict:
    """One batched sampling pass; every panel is derived from it.

    Returns the per-token tag matrix for all n trajectories, the full step-by-step
    intermediates of the first one, and the decoded triplet of each, so the
    trajectory view and the extraction share a single draw.
    """
    model, _ = load_cached_model(config_path, checkpoint_path)
    words = sentence.split()
    torch.manual_seed(seed)

    encodings = model.encoder.tokenizer(
        [words], is_split_into_words=True, add_special_tokens=True,
        padding=True, return_tensors="pt",
    )
    word_ids = encodings.word_ids(batch_index=0)
    ids = encodings["input_ids"].to(model.device)
    attn = encodings["attention_mask"].to(model.device)

    started = time.perf_counter()
    with torch.no_grad():
        context = model.encode_tokens(ids, attn)
        context = context.repeat_interleave(n, dim=0)
        attn_n = attn.repeat_interleave(n, dim=0)
        samples, intermediates = model.generate(
            batch_size=n, context=context, context_attention_mask=attn_n,
            tag_attention_mask=attn_n, return_intermediate=True,
        )
    elapsed = time.perf_counter() - started
    samples = samples.cpu()

    # Same span decode the model uses (discrete_model.get_triplets / mc_dropout_tagger).
    triplets = [
        (
            extract_longest_span(samples[i] == SEQ_STR2INT["S"], word_ids),
            extract_longest_span(samples[i] == SEQ_STR2INT["O"], word_ids),
            extract_longest_span(samples[i] == SEQ_STR2INT["R"], word_ids),
        )
        for i in range(n)
    ]
    return {
        "tokens": model.encoder.tokenizer.convert_ids_to_tokens(ids[0].tolist()),
        "word_ids": word_ids,
        "samples": samples,
        "intermediates": intermediates[0].cpu(),
        "triplets": triplets,
        "elapsed": elapsed,
        "uniform": model.scheduler.kernel == "uniform",
    }


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


def reroll() -> None:
    st.session_state.seed = random.randrange(2**31 - 1)


def main() -> None:
    cli = parse_args()
    st.set_page_config(page_title="DiffIE Demo", layout="wide", page_icon="🔬")
    pal = palette()

    st.title("DiffIE")
    st.caption(
        "Open Information Extraction as discrete diffusion — independent "
        "reverse-diffusion trajectories over per-token role tags, clustered "
        "into triplets."
    )

    st.session_state.setdefault("seed", 42)

    with st.sidebar:
        with st.expander("Model", expanded=False):
            config_path = st.text_input("Config", cli.config)
            checkpoint_path = st.text_input("Checkpoint", cli.checkpoint)

        st.subheader("Sampling")
        num_samples = st.select_slider(
            "Trajectories (n)", [1, 2, 4, 8, 16, 32, 64, 128, 256, 512], value=64,
            help="Independent reverse-diffusion trajectories. "
                 "More = better recall, slower.",
        )
        topk = st.slider("Triplets returned (k)", 1, 10, 4)
        threshold = st.slider("Lenient-match threshold (τ)", 0.5, 1.0, 0.9, 0.05)
        st.caption("k and τ re-cluster the same draw — no resampling, so they "
                   "apply instantly.")

        st.subheader("Randomness")
        st.number_input("Seed", min_value=0, step=1, key="seed")

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
    st.button("Resample", type="primary", on_click=reroll,
              help="Draw a fresh set of trajectories with a new seed.")
    if not sentence.strip():
        st.info("Enter a sentence above.")
        return

    words = sentence.split()
    seed = int(st.session_state.seed)
    with st.spinner(f"Sampling {num_samples} trajectories..."):
        drawn = draw_trajectories(config_path, checkpoint_path, sentence,
                                  int(num_samples), seed)

    tokens = drawn["tokens"]
    samples, uniform = drawn["samples"], drawn["uniform"]
    steps = drawn["intermediates"].shape[1]

    extractor = LenientFrequencyExtractor(
        k=int(num_samples), topk=int(topk), threshold=float(threshold)
    )
    triplets, masses = extractor.get_carb_prediction(
        words, lambda _batch, n=None: drawn["triplets"]
    )

    cols = st.columns(4)
    cols[0].metric("Trajectories", num_samples)
    cols[1].metric("Denoising steps", steps)
    cols[2].metric("Triplets", len(triplets))
    cols[3].metric("Sampling time", f"{drawn['elapsed']:.2f}s")

    # ── Trajectory first: how the tags actually get made ──────────────────────
    st.subheader("Denoising trajectory")
    st.caption(
        f"One of the {num_samples} trajectories. Each row is a denoising step, "
        f"from pure noise at t={steps} down to the tags the triplets are read off. "
        "Read it top to bottom."
    )
    shown_tags = ["B", "S", "R", "O"] + (["P" if uniform else "M"] if steps > 1 else [])
    final_token_tags = [fix_tag(state_id_to_str(int(s_)), uniform) for s_ in samples[0]]

    rows = [
        (
            f"t={steps - j}",
            [fix_tag(state_id_to_str(int(s_)), uniform)
             for s_ in drawn["intermediates"][:, j]],
        )
        for j in range(steps)
    ]
    rows.append(("final", final_token_tags))
    st.html(render_matrix(tokens, rows, pal))
    st.html(render_legend(pal, shown_tags))

    # ── Then the result ───────────────────────────────────────────────────────
    st.subheader("Extracted triplets")
    st.caption(
        f"All {num_samples} trajectories clustered by lenient match (τ={threshold}) "
        f"and ranked by how much of the pool each cluster holds; top {topk} shown."
    )
    if not triplets:
        st.warning("No triplets extracted — try a longer sentence or a lower τ.")
    for (sub_span, obj_span, pred_span), mass in zip(triplets, masses):
        with st.container(border=True):
            st.html(render_triplet_card(
                mass, span_text(words, sub_span), span_text(words, pred_span),
                span_text(words, obj_span), pal,
            ))
    if triplets:
        with st.expander("As a table"):
            st.dataframe(
                pd.DataFrame([
                    {"Score": round(m, 3),
                     "Subject": span_text(words, s_),
                     "Relation": span_text(words, p_),
                     "Object": span_text(words, o_)}
                    for (s_, o_, p_), m in zip(triplets, masses)
                ]),
                hide_index=True, width="stretch",
            )


if __name__ == "__main__":
    main()
