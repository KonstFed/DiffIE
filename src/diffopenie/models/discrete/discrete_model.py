from typing import Annotated, Union

from torch import nn
import torch
from pydantic import BaseModel, ConfigDict, Field

from diffopenie.models.base_model import BaseTripletModel
from diffopenie.models.encoder import BERTEncoder, BERTEncoderConfig
from diffopenie.diffusion.discrete import D3PMSchedule, D3PMScheduleConfig
from diffopenie.diffusion.mdlm import MDLMSchedule, MDLMScheduleConfig
from diffopenie.models.discrete.denoiser import DiscreteDenoiser, DiscreteDenoiserConfig
from diffopenie.data.triplet_utils import extract_longest_span
from diffopenie.data import SEQ_INT2STR, SEQ_STR2INT

SchedulerConfig = Annotated[
    Union[D3PMScheduleConfig, MDLMScheduleConfig],
    Field(discriminator="type"),
]


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
        scheduler: D3PMSchedule | MDLMSchedule,
        denoiser: DiscreteDenoiser,
        temperature: float = 1.0,
        topk: int | None = None,
        argmax: bool = False,
        use_remasking: bool = False,
        remask_threshold_low: float = 0.3,
        remask_threshold_high: float = 1.0,
    ):
        super().__init__()
        self.encoder = encoder
        self.scheduler = scheduler
        self.denoiser = denoiser
        self.temperature = temperature
        self.topk = topk
        self.argmax = argmax
        self.use_remasking = use_remasking
        self.remask_threshold_low = remask_threshold_low
        self.remask_threshold_high = remask_threshold_high

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
        argmax: bool = False,
    ) -> torch.LongTensor:
        return self.scheduler.sample_reverse(x_t, t, p_x0_given_xt, argmax)

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
        return_intermediate: bool = False,
    ) -> torch.LongTensor | tuple[torch.LongTensor, torch.LongTensor]:
        """
        Reverse diffusion sampling loop (paper Eq. 4 construction).
        Uses config: temperature/argmax, use_remasking, remask_threshold_*.

        Args:
            return_intermediate: If True, return (x_0, intermediates) where
                intermediates is [B, L, T] with predictions at each reverse step
                (intermediates[:,:,0] after t=T, ..., intermediates[:,:,T-1] = x_0).
        """
        B = batch_size
        L = token_embeddings.shape[1]
        K = self.num_states
        T = self.scheduler.num_steps
        mask_state_id = self.scheduler.mask_state_id if self.scheduler.kernel == "mask_absorbing" else None

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

        intermediates_list: list[torch.LongTensor] = [] if return_intermediate else []

        for ti in range(T, 0, -1):
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
            x_t = self.sample_reverse(x_t, t, p_x0, argmax=self.argmax).to(self.device)

            if self.use_remasking and mask_state_id is not None:
                confidence = p_x0.max(dim=-1).values  # (B, L)
                threshold = self.remask_threshold_low + (self.remask_threshold_high - self.remask_threshold_low) * (ti / T)
                remask = confidence < threshold
                # only remask within valid tokens
                remask = remask & attention_mask.to(torch.bool)
                x_t = torch.where(remask, torch.full_like(x_t, mask_state_id), x_t)

            if return_intermediate:
                intermediates_list.append(x_t.clone())

        if return_intermediate:
            # [B, L, T]: dim 2 index 0 = after step t=T, ..., T-1 = x_0
            intermediates = torch.stack(intermediates_list, dim=2)
            return x_t, intermediates
        return x_t


class DiscreteModelConfig(BaseModel):
    """
    Configuration model for DiscreteModel.
    Composes encoder, scheduler, denoiser configs; create() builds the model.
    """

    model_config = ConfigDict(extra="forbid")
    encoder: BERTEncoderConfig
    scheduler: SchedulerConfig
    denoiser: DiscreteDenoiserConfig
    temperature: float = 1.0
    topk: int | None = None
    argmax: bool = False
    use_remasking: bool = False
    remask_threshold_low: float = 0.3
    remask_threshold_high: float = 1.0

    def create(self) -> DiscreteModel:
        """Build DiscreteModel from configs."""
        return DiscreteModel(
            encoder=self.encoder.create(),
            scheduler=self.scheduler.create(),
            denoiser=self.denoiser.create(),
            temperature=self.temperature,
            topk=self.topk,
            argmax=self.argmax,
            use_remasking=self.use_remasking,
            remask_threshold_low=self.remask_threshold_low,
            remask_threshold_high=self.remask_threshold_high,
        )
