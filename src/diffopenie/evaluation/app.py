"""Interactive Streamlit debugger for DiffIE diffusion model.

Usage:
    uv run streamlit run src/diffopenie/evaluation/app.py
"""

from __future__ import annotations

import html
import io
from pathlib import Path

import matplotlib.animation as mpl_animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch

from diffopenie.data.lsoie import SequenceLSOEIDataset, labels_to_indices
from diffopenie.evaluation.debug_diffusion import (
    load_carb_data,
    load_model,
    state_id_to_str,
)
from diffopenie.data.triplet_utils import per_token_entropy
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
_LABEL_NAMES = ["B", "S", "R", "O"]
_LABEL_COLORS = [_TAG_COLORS[l][0] for l in _LABEL_NAMES]


# ── HTML helpers ───────────────────────────────────────────────────────────────


def _cell(tag: str, text: str = "") -> str:
    bg, fg = _TAG_COLORS.get(tag, ("#fff", "#000"))
    label = html.escape(text or tag)
    return (
        f'<td style="background:{bg};color:{fg};padding:2px 5px;'
        f"text-align:center;font-family:monospace;font-size:12px;"
        f'border:1px solid #ccc;white-space:nowrap">{label}</td>'
    )


def _header_cell(text: str) -> str:
    return (
        f'<th style="padding:2px 5px;font-size:11px;'
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

    cells = _row_label_cell("Step", bg="#e8e8e8")
    for tok in tokens:
        cells += _header_cell(tok)
    rows.append(f"<tr>{cells}</tr>")

    if gt_tags is not None:
        cells = _row_label_cell("GT", bg="#fffde7")
        for tag in gt_tags:
            cells += _cell(tag)
        rows.append(f"<tr>{cells}</tr>")

    for step_idx in range(T):
        ti = T - step_idx
        tags = [state_id_to_str(int(s)) for s in intermediates[:, step_idx]]
        cells = _row_label_cell(f"t={ti}", bg="#fafafa")
        for tag in tags:
            cells += _cell(tag)
        rows.append(f"<tr>{cells}</tr>")

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
            label = {"B": "Background", "S": "Subject", "R": "Relation", "O": "Object"}[tag]
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


def run_n_generations(
    model,
    token_ids: list[int],
    n: int,
    seed: int | None,
) -> torch.Tensor:
    """Encode once, generate n times. Returns [n, L] integer tensor."""
    if seed is not None:
        torch.manual_seed(seed)
    device = model.device
    ids_t = torch.tensor([token_ids], dtype=torch.long, device=device)
    attn = torch.ones_like(ids_t)
    with torch.no_grad():
        token_emb = model.encode_tokens(ids_t, attn)  # [1, L, D]
        token_emb = token_emb.repeat(n, 1, 1)         # [n, L, D]
        attn_rep = attn.repeat(n, 1)                   # [n, L]
        samples = model.generate(
            batch_size=n,
            token_embeddings=token_emb,
            attention_mask=attn_rep,
        )
    return samples.cpu()  # [n, L]


# ── Entropy analysis helpers ───────────────────────────────────────────────────


def compute_entropy_series(
    samples: torch.Tensor,  # [N, L]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Incrementally compute per-token entropy after each generation.

    Returns:
        entropy_matrix: [N, L] — H[k, l] = entropy at token l after k+1 samples
        mean_entropy:   [N]    — mean over tokens
        max_entropy:    [N]    — max over tokens
    """
    N, L = samples.shape
    entropy_matrix = np.zeros((N, L))
    mean_entropy = np.zeros(N)
    max_entropy = np.zeros(N)
    for k in range(N):
        H = per_token_entropy(
            samples[: k + 1].clamp(0, 3), num_classes=4, normalize=True
        )
        entropy_matrix[k] = H.numpy()
        mean_entropy[k] = H.mean().item()
        max_entropy[k] = H.max().item()
    return entropy_matrix, mean_entropy, max_entropy


def build_distribution_frames(samples: torch.Tensor) -> np.ndarray:
    """
    Compute cumulative empirical distribution after each generation.

    Returns: [N, L, 4] array — probs[k, l, c] = P(label=c | token=l) after k+1 samples
    """
    N, L = samples.shape
    cumulative = torch.zeros(L, 4)
    all_probs = np.zeros((N, L, 4))
    for k in range(N):
        row = samples[k].clamp(0, 3)  # [L]
        cumulative.scatter_add_(1, row.unsqueeze(1), torch.ones(L, 1))
        all_probs[k] = (cumulative / (k + 1)).numpy()
    return all_probs


# ── Plotly charts ──────────────────────────────────────────────────────────────


def plot_distribution_animation(
    all_probs: np.ndarray,  # [N, L, 4]
    token_labels: list[str],
) -> go.Figure:
    N, L, _ = all_probs.shape
    x = list(range(L))

    def make_traces(k: int) -> list[go.Bar]:
        bottoms = np.zeros(L)
        traces = []
        for c, (label, color) in enumerate(zip(_LABEL_NAMES, _LABEL_COLORS)):
            traces.append(
                go.Bar(
                    x=x,
                    y=all_probs[k, :, c],
                    base=bottoms,
                    name=label,
                    marker_color=color,
                    showlegend=(k == 0),
                    hovertemplate=f"{label}: %{{y:.2f}}<extra></extra>",
                )
            )
            bottoms = bottoms + all_probs[k, :, c]
        return traces

    fig = go.Figure(
        data=make_traces(0),
        layout=go.Layout(
            barmode="stack",
            xaxis=dict(tickmode="array", tickvals=x, ticktext=token_labels, tickangle=45),
            yaxis=dict(range=[0, 1.05], title="Probability"),
            title="Empirical label distribution",
            legend=dict(orientation="h", y=1.12),
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    y=1.15,
                    x=0,
                    xanchor="left",
                    buttons=[
                        dict(label="▶ Play", method="animate",
                             args=[None, {"frame": {"duration": 120, "redraw": True},
                                          "fromcurrent": True}]),
                        dict(label="⏸ Pause", method="animate",
                             args=[[None], {"frame": {"duration": 0}, "mode": "immediate"}]),
                    ],
                )
            ],
            sliders=[dict(
                steps=[
                    dict(method="animate", args=[[f"frame{k}"],
                                                  {"mode": "immediate",
                                                   "frame": {"duration": 120, "redraw": True}}],
                         label=str(k + 1))
                    for k in range(N)
                ],
                active=0,
                currentvalue=dict(prefix="Generation: "),
                len=1.0, x=0, y=-0.15,
            )],
        ),
        frames=[
            go.Frame(data=make_traces(k), name=f"frame{k}")
            for k in range(N)
        ],
    )
    return fig


def plot_mean_max_entropy(mean_entropy: np.ndarray, max_entropy: np.ndarray) -> go.Figure:
    n = np.arange(1, len(mean_entropy) + 1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=n, y=mean_entropy, mode="lines", name="Mean entropy",
                             line=dict(color="#4c72b0")))
    fig.add_trace(go.Scatter(x=n, y=max_entropy, mode="lines", name="Max entropy",
                             line=dict(color="#dd8452", dash="dash")))
    fig.update_layout(
        title="Sequence entropy over generations",
        xaxis_title="Generations (n)",
        yaxis_title="Normalized entropy",
        yaxis=dict(range=[0, 1.05]),
        legend=dict(orientation="h", y=1.12),
    )
    return fig


def plot_entropy_heatmap(
    entropy_matrix: np.ndarray,  # [N, L]
    token_labels: list[str],
) -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z=entropy_matrix,
        x=token_labels,
        y=np.arange(1, entropy_matrix.shape[0] + 1),
        colorscale="Reds",
        colorbar=dict(title="H (norm.)"),
        hovertemplate="gen %{y}, token %{x}: H=%{z:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title="Per-token entropy over generations",
        xaxis=dict(title="Token", tickangle=45),
        yaxis=dict(title="Generations (n)", autorange="reversed"),
    )
    return fig


# ── GIF ────────────────────────────────────────────────────────────────────────


def make_distribution_gif(
    all_probs: np.ndarray,  # [N, L, 4]
    token_labels: list[str],
    fps: int = 5,
) -> bytes:
    N, L, _ = all_probs.shape
    fig, ax = plt.subplots(figsize=(max(8, L * 0.45), 4))
    fig.tight_layout()
    x = np.arange(L)

    def update(k: int):
        ax.clear()
        bottoms = np.zeros(L)
        for c, (label, color) in enumerate(zip(_LABEL_NAMES, _LABEL_COLORS)):
            ax.bar(x, all_probs[k, :, c], bottom=bottoms, color=color,
                   label=label, width=0.8)
            bottoms += all_probs[k, :, c]
        ax.set_xticks(x)
        ax.set_xticklabels(token_labels, rotation=45, ha="right", fontsize=7)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Probability")
        ax.set_title(f"Empirical label distribution — {k + 1} generations")
        ax.legend(loc="upper right", fontsize=8, ncol=4)

    anim = mpl_animation.FuncAnimation(fig, update, frames=N, interval=200)
    buf = io.BytesIO()
    anim.save(buf, writer="pillow", fps=fps)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


# ── Pages ──────────────────────────────────────────────────────────────────────


def page_debug(model, config, tokenizer, dataset, split, index,
               custom_sentence, carb_dir_str, actual_seed) -> None:
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
    state_key = str(token_ids)
    reroll = st.button("🎲  Reroll generation", type="primary")

    need_gen = (
        reroll
        or "gen_key" not in st.session_state
        or st.session_state["gen_key"] != state_key
    )

    if need_gen:
        with st.spinner("Running diffusion..."):
            x_0, intermediates, final_tags = run_generation(model, token_ids, actual_seed)
        st.session_state.update(
            gen_key=state_key, x_0=x_0, intermediates=intermediates, final_tags=final_tags,
        )
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

    # ── Diffusion steps ───────────────────────────────────────────────────────
    st.subheader("Diffusion Steps")
    if words:
        st.markdown(f"**Sentence:** {' '.join(words)}")

    if gt_labels is not None:
        correct = (x_0 == gt_labels).sum().item()
        total = len(gt_labels)
        st.metric("Token accuracy", f"{100 * correct / total:.1f}%", f"{correct}/{total}")

    st.markdown(
        build_diffusion_table(tokens, gt_tags, intermediates, final_tags, T),
        unsafe_allow_html=True,
    )
    st.markdown(_legend_html(), unsafe_allow_html=True)

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
                    {"Tag": name, "Correct": n_correct, "Total": n_total,
                     "Acc": f"{100 * n_correct / n_total:.1f}%"}
                )
            st.dataframe(pd.DataFrame(rows_acc), hide_index=True)

    # ── Triplets ──────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Triplets")
    if words is None:
        st.info("Triplet extraction requires a sentence with word boundaries.")
    else:
        col_pred, col_gt = st.columns(2)
        with col_pred:
            st.markdown("**Predicted (this roll)**")
            pred_s, pred_r, pred_o = _tags_to_token_texts(final_tags, tokens)
            st.markdown(f"**Subject:** {pred_s}  \n**Relation:** {pred_r}  \n**Object:** {pred_o}")

        with col_gt:
            st.markdown("**Ground Truth**")
            if all_gt_lsoie:
                rows_gt = []
                for labels in all_gt_lsoie:
                    sub_span, obj_span, pred_span = labels_to_indices(labels)
                    rows_gt.append(
                        {"Subject": _span_text(words, sub_span),
                         "Relation": _span_text(words, pred_span),
                         "Object": _span_text(words, obj_span)}
                    )
                st.dataframe(pd.DataFrame(rows_gt), hide_index=True, use_container_width=True)
            elif all_gt_carb:
                rows_gt = [{"Subject": s, "Relation": p, "Object": o} for s, p, o in all_gt_carb]
                st.dataframe(pd.DataFrame(rows_gt), hide_index=True, use_container_width=True)
            else:
                st.info("No ground truth available.")

    # ── CaRB prediction ───────────────────────────────────────────────────────
    st.divider()
    st.subheader("CaRB Prediction")
    if words is None:
        st.info("CaRB prediction requires a sentence.")
    else:
        st.caption("Samples model multiple times and returns the most frequent triplets.")
        run_carb = st.button("▶  Run CaRB prediction")

        if run_carb:
            with st.spinner("Sampling..."):
                triplets, probs = model.get_carb_prediction(words)
            st.session_state["carb_results"] = (triplets, probs, list(words))

        if "carb_results" in st.session_state:
            cached_triplets, cached_probs, cached_words = st.session_state["carb_results"]
            pred_rows = []
            for (sub_span, obj_span, pred_span), prob in zip(cached_triplets, cached_probs):
                pred_rows.append(
                    {"Prob": f"{prob:.3f}",
                     "Subject": _span_text(cached_words, sub_span),
                     "Relation": _span_text(cached_words, pred_span),
                     "Object": _span_text(cached_words, obj_span)}
                )
            if pred_rows:
                st.dataframe(pd.DataFrame(pred_rows), hide_index=True, use_container_width=True)
            else:
                st.warning("No triplets extracted.")

            if all_gt_carb or all_gt_lsoie:
                st.markdown("**Gold triplets**")
                if all_gt_carb:
                    gold_rows = [{"Subject": s, "Relation": p, "Object": o}
                                 for s, p, o in all_gt_carb]
                else:
                    gold_rows = []
                    for labels in all_gt_lsoie:  # type: ignore[union-attr]
                        sub_span, obj_span, pred_span = labels_to_indices(labels)
                        gold_rows.append(
                            {"Subject": _span_text(cached_words, sub_span),
                             "Relation": _span_text(cached_words, pred_span),
                             "Object": _span_text(cached_words, obj_span)}
                        )
                st.dataframe(
                    pd.DataFrame(gold_rows), hide_index=True, use_container_width=True
                )


def page_entropy(model, config, tokenizer, dataset, split, index,
                 custom_sentence, carb_dir_str, actual_seed) -> None:
    # ── Load token IDs ────────────────────────────────────────────────────────
    try:
        tokens: list[str] = []
        token_ids: list[int] = []

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
            sentences, _ = load_carb_data(
                carb_path / "data" / f"{sp}.txt",
                carb_path / "data" / "gold" / f"{sp}.tsv",
            )
            words = sentences[index].split()
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

    except Exception as exc:
        st.error(f"Failed to load example: {exc}")
        return

    st.markdown(f"**Tokens ({len(tokens)}):** {' '.join(tokens)}")

    # ── Controls ──────────────────────────────────────────────────────────────
    n_gen = st.number_input("Number of generations (N)", min_value=2, max_value=2000,
                            value=50, step=10)
    run_btn = st.button("▶  Run N generations", type="primary")

    state_key = f"{token_ids}_{n_gen}"
    if run_btn or st.session_state.get("entropy_key") != state_key:
        if run_btn:
            with st.spinner(f"Generating {n_gen} samples..."):
                samples = run_n_generations(model, token_ids, int(n_gen), actual_seed)
            st.session_state["entropy_samples"] = samples
            st.session_state["entropy_key"] = state_key

    if "entropy_samples" not in st.session_state:
        st.info("Click **Run N generations** to start.")
        return

    samples: torch.Tensor = st.session_state["entropy_samples"]

    with st.spinner("Computing entropy..."):
        all_probs = build_distribution_frames(samples)
        entropy_matrix, mean_entropy, max_entropy = compute_entropy_series(samples)

    # ── Charts ────────────────────────────────────────────────────────────────
    col_dist, col_entropy = st.columns(2)

    with col_dist:
        st.plotly_chart(
            plot_distribution_animation(all_probs, tokens),
            use_container_width=True,
        )

    with col_entropy:
        st.plotly_chart(
            plot_mean_max_entropy(mean_entropy, max_entropy),
            use_container_width=True,
        )

    st.plotly_chart(
        plot_entropy_heatmap(entropy_matrix, tokens),
        use_container_width=True,
    )

    # ── GIF download ──────────────────────────────────────────────────────────
    if st.button("🎞  Build GIF (slow)"):
        with st.spinner("Rendering GIF..."):
            gif_bytes = make_distribution_gif(all_probs, tokens)
        st.download_button(
            label="⬇  Download distribution GIF",
            data=gif_bytes,
            file_name="label_distribution.gif",
            mime="image/gif",
        )


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    st.set_page_config(page_title="DiffIE Debugger", layout="wide", page_icon="🔬")
    st.title("DiffIE — Interactive Diffusion Debugger")

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Model")
        config_path = st.text_input("Config path", "config/training_mdlm.yaml")
        checkpoint_path = st.text_input("Checkpoint path", "checkpoints/checkpoint_best.pt")
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

        st.divider()
        page = st.radio("Page", ["🔬 Debug", "📊 Entropy Analysis"])

    # ── Model loading ─────────────────────────────────────────────────────────
    if load_btn:
        _load_model_cached.clear()

    if not config_path or not checkpoint_path:
        st.info("Enter config and checkpoint paths in the sidebar, then click **Load Model**.")
        return

    try:
        with st.spinner("Loading model..."):
            model, config = _load_model_cached(config_path, checkpoint_path)
    except Exception as exc:
        st.error(f"Failed to load model: {exc}")
        return

    tokenizer = model.encoder.tokenizer

    # ── Dispatch ──────────────────────────────────────────────────────────────
    kwargs = dict(
        model=model, config=config, tokenizer=tokenizer,
        dataset=dataset, split=split, index=index,
        custom_sentence=custom_sentence, carb_dir_str=carb_dir_str,
        actual_seed=actual_seed,
    )
    if page == "🔬 Debug":
        page_debug(**kwargs)
    else:
        page_entropy(**kwargs)


if __name__ == "__main__":
    main()
