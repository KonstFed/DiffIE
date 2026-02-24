from torch import nn
import torch
from pydantic import BaseModel

from diffopenie.models.base_model import BaseTripletModel
from diffopenie.models.encoder import BERTEncoder, BERTEncoderConfig
from diffopenie.diffusion.discrete import D3PMSchedule, D3PMScheduleConfig
from diffopenie.models.discrete.denoiser import DiscreteDenoiser, DiscreteDenoiserConfig
from diffopenie.data.triplet_utils import extract_longest_span
from diffopenie.data import SEQ_INT2STR, SEQ_STR2INT


def _topk_filter_logits(logits: torch.Tensor, k: int) -> torch.Tensor:
    """
    Keep only top-k logits (set others to -inf), per row.
    logits: (..., K)
    """
    if k <= 0 or k >= logits.size(-1):
        return logits
    vals, _ = torch.topk(logits, k, dim=-1)
    thresh = vals[..., -1].unsqueeze(-1)
    return torch.where(logits >= thresh, logits, torch.full_like(logits, float("-inf")))


class DiscreteModel(nn.Module, BaseTripletModel):
    def __init__(
        self,
        encoder: BERTEncoder,
        scheduler: D3PMSchedule,
        denoiser: DiscreteDenoiser,
        temperature: float = 1.0,
        topk: int | None = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.scheduler = scheduler
        self.denoiser = denoiser
        self.temperature = temperature
        self.topk = topk

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def num_states(self) -> int:
        return self.scheduler.num_states

    def sample_reverse(
        self,
        x_t: torch.LongTensor,
        t: torch.LongTensor,
        p_x0_given_xt: torch.Tensor,
    ) -> torch.LongTensor:
        return self.scheduler.sample_reverse(x_t, t, p_x0_given_xt)

    # scheduler wrappers
    def noise(self, x0: torch.LongTensor, t: torch.LongTensor) -> torch.LongTensor:
        return self.scheduler.sample_forward(x0, t)

    def denoise(
        self,
        x_t: torch.LongTensor,
        t: torch.LongTensor,
        token_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.LongTensor:
        return self.denoiser(x_t, t, token_embeddings, attention_mask)

    # inference below
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

    @torch.no_grad()
    def get_triplets(
        self, words: list[list[str]]
    ) -> list[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]]:
        """
        Get triplets (subj_span, obj_span, pred_span) as word index spans from a
        batch of word lists. Uses generate() for reverse diffusion, then decodes
        state indices (1=subj, 2=obj, 3=pred) to spans.
        """
        if not words:
            return []
        device = self.device
        encodings = self.encoder.tokenizer(
            words,
            is_split_into_words=True,
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
        )
        token_ids = encodings["input_ids"].to(device)
        attention_mask = encodings["attention_mask"].to(device)
        token_embeddings = self.encode_tokens(token_ids, attention_mask)
        batch_size = token_embeddings.shape[0]
        pred_states = self.generate(
            batch_size=batch_size,
            token_embeddings=token_embeddings,
            attention_mask=attention_mask,
        )
        pred_states = pred_states.cpu()
        results = []
        for i in range(len(words)):
            word_ids = encodings.word_ids(batch_index=i)
            sub_span = extract_longest_span((pred_states[i] == SEQ_STR2INT["S"]), word_ids)
            obj_span = extract_longest_span((pred_states[i] == SEQ_STR2INT["O"]), word_ids)
            pred_span = extract_longest_span((pred_states[i] == SEQ_STR2INT["R"]), word_ids)
            results.append(
                (
                    sub_span if sub_span is not None else (0, 0),
                    obj_span if obj_span is not None else (0, 0),
                    pred_span if pred_span is not None else (0, 0),
                )
            )
        return results

    @torch.no_grad()
    def generate(
        self,
        *,
        batch_size: int,
        token_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.LongTensor:
        """
        Reverse diffusion sampling loop (paper Eq. 4 construction).

        For t = T..1:
        1) Predict p_θ(x0 | x_t, t) with the denoiser:
            logits = denoiser(x_t, t, **condition)              # (B, L, K)
            p_x0   = softmax(logits / temperature)

        2) Form reverse transition distribution (paper Eq. 4, matrix form):
            p_θ(x_{t-1} | x_t) ∝ (x_t Q_t^T) ⊙ (p_θ(x0|x_t) \bar Q_{t-1})

        3) Sample x_{t-1} ~ p_θ(x_{t-1} | x_t)

        Initialization (x_T):
        - kernel == "mask_absorbing": x_T := MASK everywhere
        - kernel == "uniform":        x_T ~ Uniform({0..K-1}) i.i.d.

        Args:
            batch_size: batch size
            token_embeddings: (B, L, ctx_dim) token embeddings
            attention_mask: (B, L) attention mask

        Returns:
            x0_sample: (B, L) sampled clean states
        """
        B = batch_size
        L = token_embeddings.shape[1]
        K = self.num_states

        # Initialize x_T
        if self.scheduler.kernel == "mask_absorbing":
            x_t = torch.full(
                (B, L),
                self.scheduler.mask_state_id,
                device=self.device,
                dtype=torch.long,
            )
        else:
            x_t = torch.randint(0, K, (B, L), device=self.device, dtype=torch.long)

        for ti in range(self.scheduler.num_steps, 0, -1):
            t = torch.full((B,), ti, device=self.device, dtype=torch.long)

            logits = self.denoiser(
                x_t, t, token_embeddings, attention_mask
            )  # (B, L, K)
            if logits.shape != (B, L, K):
                raise ValueError(f"denoiser must return logits of shape {(B, L, K)}")

            if self.temperature != 1.0:
                logits = logits / max(self.temperature, 1e-8)

            # if self.topk is not None:
            #     logits = _topk_filter_logits(logits, self.topk)

            p_x0 = torch.softmax(logits, dim=-1)
            x_t = self.sample_reverse(x_t, t, p_x0).to(self.device)

        # check if are there any mask DEBUG
        if self.scheduler.kernel == "mask_absorbing":
            mask_state_id = self.scheduler.mask_state_id
            for sample_ind in range(batch_size):
                if (x_t[sample_ind] == mask_state_id).any():
                    l = attention_mask[sample_ind].sum()
                    mask_number = (x_t == mask_state_id).sum()
                    print("AAAAAAAAA", mask_number, l, mask_number / l)
        return x_t

    # pretty utils
    @staticmethod
    def decode_predictions(predictions: torch.Tensor) -> list[str]:
        """Decode predictions to text.

        Args:
            predictions (torch.Tensor): [L] single prediction

        Returns:
            list[str]: tags
        """
        ind2str = {
            "0": "<BOS>",
            "1": "<SUBJ>",
            "2": "<OBJ>",
            "3": "<PRED>",
            "4": "<MASK>",
        }
        return [ind2str[str(p)] for p in predictions]


class DiscreteModelConfig(BaseModel):
    """
    Configuration model for DiscreteModel.
    Composes encoder, scheduler, denoiser configs; create() builds the model.
    """

    encoder: BERTEncoderConfig
    scheduler: D3PMScheduleConfig
    denoiser: DiscreteDenoiserConfig
    temperature: float = 1.0
    topk: int | None = None

    def create(self) -> DiscreteModel:
        """Build DiscreteModel from configs."""
        return DiscreteModel(
            encoder=self.encoder.create(),
            scheduler=self.scheduler.create(),
            denoiser=self.denoiser.create(),
            temperature=self.temperature,
            topk=self.topk,
        )
