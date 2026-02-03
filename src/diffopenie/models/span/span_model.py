from torch import nn
import torch
from pydantic import BaseModel

from diffopenie.models.base_model import BaseTripletModel
from diffopenie.models.encoder import BERTEncoder, BERTEncoderConfig
from diffopenie.models.span.label_mapper import ContinuousSpanMapper
from diffopenie.models.span.denoiser import SpanDenoiser
from diffopenie.diffusion.scheduler import LinearScheduler, LinearSchedulerConfig

def spans_to_token_labels(
    spans: torch.LongTensor, seq_len: int
) -> torch.LongTensor:
    """[B, 6] span indices -> [B, L] token labels (0=O, 1=subj, 2=obj, 3=pred)."""
    B = spans.shape[0]
    device = spans.device
    out = torch.zeros(B, seq_len, dtype=torch.long, device=device)
    for b in range(B):
        s_l, s_r = max(0, spans[b, 0].item()), min(seq_len - 1, spans[b, 1].item())
        o_l, o_r = max(0, spans[b, 2].item()), min(seq_len - 1, spans[b, 3].item())
        p_l, p_r = max(0, spans[b, 4].item()), min(seq_len - 1, spans[b, 5].item())
        if s_l <= s_r:
            out[b, s_l : s_r + 1] = 1
        if o_l <= o_r:
            out[b, o_l : o_r + 1] = 2
        if p_l <= p_r:
            out[b, p_l : p_r + 1] = 3
    return out


class SpanDiffusionModel(nn.Module, BaseTripletModel):
    """
    Unified diffusion model for span labeling.
    """

    def __init__(
        self,
        denoiser: SpanDenoiser,
        scheduler: LinearScheduler,
        label_mapper: ContinuousSpanMapper,
        encoder: BERTEncoder,
    ):
        super().__init__()
        self.denoiser = denoiser
        self.scheduler = scheduler
        self.label_mapper = label_mapper
        self.encoder = encoder

    @torch.no_grad()
    def get_triplets(
        self, words: list[list[str]]
    ) -> list[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]]:
        """Get triplets (subj, obj, pred) as word spans from a batch of word lists."""
        # TODO: Vibe coded, need verification
        if not words:
            return []
        device = next(self.parameters()).device
        encodings = self.encoder.tokenizer(
            words,
            is_split_into_words=True,
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
        )
        token_ids = encodings["input_ids"].to(device)
        attention_mask = encodings["attention_mask"].to(device)
        pred_spans = self.predict(token_ids, attention_mask).cpu()  # [B, 6]
        num_words = [len(w) for w in words]
        results = []
        for i in range(len(words)):
            word_ids_list = encodings.word_ids(batch_index=i)
            L = len(word_ids_list)
            nw = num_words[i]

            def token_to_word(tok_idx: int, default: int) -> int:
                idx = max(0, min(int(tok_idx), L - 1))
                w = word_ids_list[idx]
                return w if w is not None else default

            s_l = token_to_word(pred_spans[i, 0].item(), 0)
            s_r = token_to_word(pred_spans[i, 1].item(), 0)
            o_l = token_to_word(pred_spans[i, 2].item(), 0)
            o_r = token_to_word(pred_spans[i, 3].item(), 0)
            p_l = token_to_word(pred_spans[i, 4].item(), 0)
            p_r = token_to_word(pred_spans[i, 5].item(), 0)
            s_l, s_r = max(0, min(s_l, nw - 1)), max(0, min(s_r, nw - 1))
            o_l, o_r = max(0, min(o_l, nw - 1)), max(0, min(o_r, nw - 1))
            p_l, p_r = max(0, min(p_l, nw - 1)), max(0, min(p_r, nw - 1))
            if s_l > s_r:
                s_r = s_l
            if o_l > o_r:
                o_r = o_l
            if p_l > p_r:
                p_r = p_l
            results.append(
                ((s_l, s_r), (o_l, o_r), (p_l, p_r))
            )
        return results

    @torch.no_grad()
    def predict(
        self,
        token_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
    ) -> torch.LongTensor:
        """Predict span indices [B, 6] from token_ids/attention_mask."""
        B, L = token_ids.shape
        device = token_ids.device
        token_embeddings = self.encode_tokens(token_ids, attention_mask)
        attn_mask = attention_mask.bool()
        condition = (token_embeddings, attn_mask)

        x_t = torch.randn(B, 6, L, device=device)
        for t_step in range(self.scheduler.num_steps - 1, -1, -1):
            x_t = x_t - x_t.mean(dim=-1, keepdim=True)
            t = torch.full((B,), t_step, dtype=torch.long, device=device)
            x_t = self.scheduler.p_sample(self.denoiser, x_t, t, condition)
        pred_spans = x_t.argmax(dim=-1)
        return pred_spans

    def noise(self, x_0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """a.k.a. forward process

        spans - [B, 6, L]"""
        # apply normal noise as usual
        noise = torch.randn_like(x_0)
        x_t = self.scheduler.q_sample(x_0, t, noise)

        # Project to zero-mean since softmax is invariant to adding a constant
        # (i.e. shifts in the all-ones direction do not change probabilities)
        x_t = x_t - x_t.mean(dim=-1, keepdim=True)
        return x_t

    def denoise(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        token_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """a.k.a. reverse process"""
        # TODO: maybe in future apply some tricks over denoised probabilities
        x_o_pred =  self.denoiser.forward(
            x_t=x_t,
            t=t,
            condition=(token_embeddings, attention_mask),
        ) # [B, 6, L]

        # normalize logits to improve training stability
        x_o_pred = x_o_pred - x_o_pred.mean(dim=-1, keepdim=True)
        x_o_pred = x_o_pred / x_o_pred.std(dim=-1, keepdim=True)
        return x_o_pred

    def encode_tokens(
        self,
        token_ids: torch.LongTensor,  # [B, L]
        attention_mask: torch.Tensor,  # [B, L]
    ) -> torch.Tensor:
        """
        Encode tokens using the BERT encoder.

        Args:
            token_ids: Token IDs [B, L]
            attention_mask: Attention mask [B, L]

        Returns:
            Token embeddings [B, L, bert_dim]
        """
        return self.encoder(token_ids, attention_mask)


# ----------------------- Pydantic configs -------------------------------------


class ContinuousSpanMapperConfig(BaseModel):
    """Config for ContinuousSpanMapper (span indices -> logits)."""

    logit_value: float = 1.0

    def create(self) -> ContinuousSpanMapper:
        return ContinuousSpanMapper(logit_value=self.logit_value)


class SpanDenoiserConfig(BaseModel):
    """Config for SpanDenoiser (slot decoder + pointer)."""

    bert_dim: int = 768
    num_steps: int = 1000
    d_model: int | None = None
    n_heads: int = 8
    n_layers: int = 1
    d_ff: int | None = None
    dropout: float = 0.1

    def create(self) -> SpanDenoiser:
        return SpanDenoiser(
            bert_dim=self.bert_dim,
            num_steps=self.num_steps,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            d_ff=self.d_ff,
            dropout=self.dropout,
        )


class SpanDiffusionModelConfig(BaseModel):
    """Config for SpanDiffusionModel (encoder + label_mapper + scheduler + denoiser)."""

    encoder: BERTEncoderConfig
    label_mapper: ContinuousSpanMapperConfig
    scheduler: LinearSchedulerConfig
    denoiser: SpanDenoiserConfig

    def create(self) -> SpanDiffusionModel:
        encoder_instance = self.encoder.create()
        label_mapper_instance = self.label_mapper.create()
        scheduler_instance = self.scheduler.create()
        denoiser_instance = self.denoiser.create()
        return SpanDiffusionModel(
            denoiser=denoiser_instance,
            scheduler=scheduler_instance,
            label_mapper=label_mapper_instance,
            encoder=encoder_instance,
        )
