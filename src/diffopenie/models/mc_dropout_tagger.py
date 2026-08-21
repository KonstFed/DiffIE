"""Non-diffusion MC-dropout BERT tagger baseline.

Isolates the contribution of discrete diffusion by replacing the reverse-diffusion
sampler with full-encoder MC-dropout over a plain BERT token classifier, while
keeping the *identical* encoder, B/S/R/O label space, span decoding, and
clustering/ranking pipeline used by ``DiscreteModel``.

The tagger deliberately mirrors ``DiscreteModel``'s method surface
(``encode_tokens``/``noise``/``denoise``/``generate``/``get_triplets``/
``get_carb_prediction`` and a ``scheduler`` exposing ``num_states``/``num_steps``)
so the existing ``Trainer`` and ``carb_eval`` reuse it unchanged. Conceptually it
is "discrete diffusion with T=1 and an identity forward process": training is plain
masked cross-entropy; stochasticity appears only at inference, where the encoder is
run in train() mode ``n`` times so dropout perturbs every layer.
"""

from typing import Literal

import torch
from torch import nn
from pydantic import BaseModel, ConfigDict, Field

from diffopenie.models.encoder import BERTEncoder, BERTEncoderConfig
from diffopenie.models.discrete.extractors import (
    ExtractorConfig,
    FrequencyExtractorConfig,
    Span,
    SpanEmbs,
    Triplet,
)
from diffopenie.models.discrete.discrete_model import _avg_span_emb
from diffopenie.data.triplet_utils import extract_longest_span
from diffopenie.data import SEQ_STR2INT

NUM_TAG_STATES = 4  # B, S, R, O — no MASK state (non-diffusion)


class _IdentityScheduler:
    """Minimal scheduler shim so the diffusion ``Trainer`` accepts the tagger.

    Presents just the surface ``Trainer`` touches. ``num_steps=1`` collapses the
    per-timestep bookkeeping to a single bucket; ``sample_t`` returns 1-indexed
    ones (``PerTimestepLoss`` subtracts 1 → index 0). It is *not* ``mask_absorbing``
    and deliberately exposes no ``weight`` attribute, so ``Trainer.compute_loss``
    skips every diffusion-only branch.
    """

    num_states: int = NUM_TAG_STATES
    num_steps: int = 1
    kernel: str = "uniform"

    def __init__(self) -> None:
        self.device = "cpu"

    def to(self, device) -> "_IdentityScheduler":
        self.device = device
        return self

    def sample_t(self, batch_size: int) -> torch.LongTensor:
        return torch.ones(batch_size, dtype=torch.long, device=self.device)


class MCDropoutTagger(nn.Module):
    def __init__(
        self,
        encoder: BERTEncoder,
        extractor,
        dropout: float = 0.3,
        temperature: float = 1.0,
        argmax: bool = True,
        inference_chunk_size: int = 128,
    ):
        super().__init__()
        self.encoder = encoder
        self.extractor = extractor
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(encoder.bert_dim, NUM_TAG_STATES)
        # `denoiser` is present only so Trainer's isinstance(denoiser, DiscreteDenoiser)
        # guard evaluates False and its grouped-batch check is skipped.
        self.denoiser = nn.Identity()
        self.scheduler = _IdentityScheduler()
        self.temperature = temperature
        self.argmax = argmax
        self.inference_chunk_size = inference_chunk_size

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def num_states(self) -> int:
        return NUM_TAG_STATES

    # -- surface used by Trainer.compute_loss -----------------------------

    def encode_tokens(
        self,
        token_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.encoder(token_ids, attention_mask)

    def noise(self, x0: torch.LongTensor, t: torch.LongTensor) -> torch.LongTensor:
        """Identity forward process — the tagger has no noising."""
        return x0

    def denoise(
        self,
        x_t: torch.LongTensor,
        t: torch.LongTensor,
        context: torch.Tensor,  # [B, L, D] BERT embeddings
        context_attention_mask: torch.Tensor,
        tag_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Per-token B/S/R/O logits from the BERT context; ignores x_t/t."""
        return self.head(self.dropout(context))

    # -- surface used by Trainer.validate (LSOIE token-overlap) -----------

    @torch.no_grad()
    def generate(
        self,
        *,
        batch_size: int,
        context: torch.Tensor,
        context_attention_mask: torch.Tensor,
        tag_attention_mask: torch.Tensor,
        return_intermediate: bool = False,
    ) -> torch.LongTensor | tuple[torch.LongTensor, torch.LongTensor]:
        """Single deterministic argmax pass over the tag positions."""
        L = tag_attention_mask.shape[1]
        logits = self.head(context)[:, :L, :]
        preds = logits.argmax(dim=-1)
        if return_intermediate:
            return preds, preds.unsqueeze(-1)  # [B, L, T=1]
        return preds

    # -- inference: MC-dropout candidate pool -----------------------------

    @torch.no_grad()
    def get_triplets(
        self,
        words: list[list[str]],
        *,
        n: int = 1,
        return_span_embs: bool = False,
    ) -> list[Triplet] | tuple[list[Triplet], list[SpanEmbs]]:
        """Full-encoder MC-dropout candidate generation.

        Runs the BERT encoder in train() mode ``n`` times per sentence (dropout
        active on every layer) so the ``n`` passes disagree, decodes each to a
        triplet via the same longest-span heuristic as ``DiscreteModel``, and
        returns ``len(words) * n`` candidates in order [n for words[0], ...].
        """
        if not words:
            return ([], []) if return_span_embs else []

        device = self.device
        encodings = self.encoder.tokenizer(
            words,
            is_split_into_words=True,
            add_special_tokens=True,
            padding=True,
            return_tensors="pt",
        )
        input_ids = encodings["input_ids"].to(device)
        attention_mask = encodings["attention_mask"].to(device)

        # Repeat each sentence n times, then run the encoder+head with dropout ON.
        rep_ids = input_ids.repeat_interleave(n, dim=0)
        rep_mask = attention_mask.repeat_interleave(n, dim=0)
        total = rep_ids.shape[0]

        was_training = self.training
        self.train()  # activate dropout in encoder + head
        try:
            states_chunks: list[torch.Tensor] = []
            embs_chunks: list[torch.Tensor] = []
            for start in range(0, total, self.inference_chunk_size):
                end = start + self.inference_chunk_size
                ctx = self.encoder(rep_ids[start:end], rep_mask[start:end])
                logits = self.head(self.dropout(ctx))
                if self.temperature != 1.0:
                    logits = logits / max(self.temperature, 1e-8)
                if self.argmax:
                    states = logits.argmax(dim=-1)
                else:
                    probs = torch.softmax(logits, dim=-1)
                    B_, L_, K_ = probs.shape
                    states = torch.multinomial(
                        probs.reshape(-1, K_), num_samples=1
                    ).reshape(B_, L_)
                states_chunks.append(states.cpu())
                if return_span_embs:
                    embs_chunks.append(ctx.cpu())
        finally:
            self.train(was_training)

        pred_states = torch.cat(states_chunks, dim=0)
        embs_cpu = torch.cat(embs_chunks, dim=0) if return_span_embs else None

        results: list[Triplet] = []
        span_embs: list[SpanEmbs] = []
        for i in range(len(words) * n):
            word_ids = encodings.word_ids(batch_index=i // n)
            sub_span = extract_longest_span(pred_states[i] == SEQ_STR2INT["S"], word_ids)
            obj_span = extract_longest_span(pred_states[i] == SEQ_STR2INT["O"], word_ids)
            pred_span = extract_longest_span(pred_states[i] == SEQ_STR2INT["R"], word_ids)
            results.append((sub_span, obj_span, pred_span))
            if return_span_embs:
                span_embs.append(
                    (
                        _avg_span_emb(embs_cpu[i], word_ids, sub_span),
                        _avg_span_emb(embs_cpu[i], word_ids, obj_span),
                        _avg_span_emb(embs_cpu[i], word_ids, pred_span),
                    )
                )

        if return_span_embs:
            return results, span_embs
        return results

    def get_carb_prediction(
        self,
        words: list[str],
    ) -> tuple[list[Triplet], list[float]]:
        """CaRB predictions via the configured extractor (identical to DiscreteModel)."""
        return self.extractor.get_carb_prediction(words, self.get_triplets)


class MCDropoutTaggerConfig(BaseModel):
    """Config for the MC-dropout tagger baseline."""

    model_config = ConfigDict(extra="forbid")
    type: Literal["mc_dropout_tagger"] = "mc_dropout_tagger"
    encoder: BERTEncoderConfig
    extractor: ExtractorConfig = Field(default_factory=FrequencyExtractorConfig)
    dropout: float = 0.3
    temperature: float = 1.0
    argmax: bool = True
    inference_chunk_size: int = 128

    def create(self) -> MCDropoutTagger:
        return MCDropoutTagger(
            encoder=self.encoder.create(),
            extractor=self.extractor.create(),
            dropout=self.dropout,
            temperature=self.temperature,
            argmax=self.argmax,
            inference_chunk_size=self.inference_chunk_size,
        )
