import math
import torch

# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------

def to_one_hot(token_ids: torch.LongTensor, num_states: int) -> torch.Tensor:
    """
    Convert token ids to row one-hot vectors.

    Args:
        token_ids: (B, L) integers in {0..num_states-1}
        num_states: K

    Returns:
        one_hot: (B, L, K) float, where one_hot[b,l,:] is a row one-hot vector.
    """
    return torch.nn.functional.one_hot(token_ids, num_classes=num_states).float()


def sample_categorical(probs: torch.Tensor) -> torch.LongTensor:
    """
    Sample from a categorical distribution row-wise.

    Args:
        probs: (..., K) nonnegative and rows sum to 1.

    Returns:
        samples: (...) integer ids in {0..K-1}.
    """
    K = probs.size(-1)
    flat = probs.reshape(-1, K)
    idx = torch.multinomial(flat, 1).squeeze(-1)
    return idx.reshape(probs.shape[:-1])


# ------------------------------------------------------------
# Beta schedules (for use with DiscreteDiffusionMarkovSchedule)
# ------------------------------------------------------------


class CosineBetaSchedule:
    """
    Cosine noise schedule for discrete diffusion (Nichol & Dhariwal style).

    Produces per-step betas from:
        alpha_bar(t) = cos^2((t + s) / (1 + s) * pi/2)
    with t in [0, 1], then beta_t = 1 - (alpha_bar_t / alpha_bar_{t-1}).

    Returns:
        betas: (num_steps,) tensor for DiscreteDiffusionMarkovSchedule(..., betas=...).
    """

    def __init__(
        self,
        num_steps: int,
        s: float = 0.008,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.num_steps = num_steps
        self.s = s
        self.device = device
        self.dtype = dtype

    def get_betas(self) -> torch.Tensor:
        steps = torch.arange(1, self.num_steps + 1, device=self.device, dtype=self.dtype)
        t = steps / self.num_steps
        alpha_bar = torch.cos((t + self.s) / (1.0 + self.s) * (math.pi / 2.0)) ** 2
        alpha_bar_prev = torch.cat(
            [torch.tensor([1.0], device=self.device, dtype=self.dtype), alpha_bar[:-1]]
        )
        alpha = alpha_bar / alpha_bar_prev
        betas = (1.0 - alpha).clamp(1e-6, 0.999)
        return betas


# ------------------------------------------------------------
# D3PM Scheduler (small-K, dense matrices, paper-faithful)
# ------------------------------------------------------------


class DiscreteDiffusionMarkovSchedule:
    """
    Discretized Denoising Diffusion Probabilistic Model (D3PM) schedule for SMALL state spaces.

    This class implements the exact *paper-style* Markov diffusion on a discrete state space:
        states = {0, 1, ..., K-1}

    Forward diffusion is a Markov chain parameterized by transition matrices:
        Q_t ∈ R^{K×K},  rows sum to 1

    Notes:
    - Time t is 1-indexed in the paper; in code, Q[0] = Q_1 and barQ[0] = I.
    - Default betas use CosineBetaSchedule when betas=None; pass betas explicitly for linear.
    """

    def __init__(
        self,
        num_states: int,
        num_steps: int,
        kernel: str = "uniform",
        mask_state_id: int | None = None,
        betas: torch.Tensor | None = None,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.num_states = num_states
        self.num_steps = num_steps
        self.kernel = kernel
        self.mask_state_id = mask_state_id
        self.device = device
        self.dtype = dtype

        if betas is None:
            cosine = CosineBetaSchedule(
                num_steps=self.num_steps, device=self.device, dtype=self.dtype
            )
            self.betas = cosine.get_betas()
        else:
            self.betas = betas.to(self.device, self.dtype)
            assert self.betas.shape == (self.num_steps,)
        self.betas = self.betas.clamp(1e-6, 0.999)

        if self.kernel == "mask_absorbing":
            if self.mask_state_id is None:
                raise ValueError("mask_state_id is required for kernel='mask_absorbing'")
            if not (0 <= self.mask_state_id < self.num_states):
                raise ValueError("mask_state_id must be in [0, num_states)")

        self.forward_transition = self._build_forward_transition_matrices()  # (T, K, K): Q_t
        self.forward_product = self._build_cumulative_products(
            self.forward_transition
        )  # (T+1, K, K): \bar Q_t

    # ----------------------------
    # Kernel construction (Q_t)
    # ----------------------------

    def _build_forward_transition_matrices(self) -> torch.Tensor:
        """
        Construct Q_t for each timestep t=1..T (stored at index t-1).

        Kernels:
        1) "uniform":
                Q_t = (1 - β_t) I + β_t U
            where U is the uniform matrix U_ij = 1/K

        2) "mask_absorbing":
            Absorbing state m = mask_state_id:
                for i != m:  P(i→i) = 1-β_t,  P(i→m) = β_t
                for i = m :  P(m→m) = 1
        """
        K = self.num_states
        I = torch.eye(K, device=self.device, dtype=self.dtype)
        U = torch.full((K, K), 1.0 / K, device=self.device, dtype=self.dtype)

        Qs = []
        for t in range(self.num_steps):
            beta = self.betas[t]
            if self.kernel == "uniform":
                Q_t = (1.0 - beta) * I + beta * U

            elif self.kernel == "mask_absorbing":
                m = self.mask_state_id
                Q_t = torch.zeros((K, K), device=self.device, dtype=self.dtype)
                Q_t += (1.0 - beta) * I
                Q_t[:, m] += beta          # leak probability into mask column
                Q_t[m, :] = 0.0
                Q_t[m, m] = 1.0            # absorbing row

            else:
                raise ValueError(f"Unknown kernel='{self.kernel}'")
            Qs.append(Q_t)

        return torch.stack(Qs, dim=0)  # (T, K, K)

    def _build_cumulative_products(self, Q: torch.Tensor) -> torch.Tensor:
        """
        Build cumulative products:
            \bar Q_0 = I
            \bar Q_t = Q_1 Q_2 ... Q_t

        Returns:
            forward_product: (T+1, K, K) where forward_product[t] = \bar Q_t
        """
        K = self.num_states
        bar = [torch.eye(K, device=self.device, dtype=self.dtype)]
        cur = bar[0]
        for t in range(self.num_steps):
            cur = cur @ Q[t]
            bar.append(cur)
        return torch.stack(bar, dim=0)

    # ----------------------------
    # Forward: q(x_t | x_0)
    # ----------------------------

    @torch.no_grad()
    def forward_distribution(self, x0: torch.LongTensor, t: torch.LongTensor) -> torch.Tensor:
        """
        Compute q(x_t | x_0) as a categorical distribution, using:
            q(x_t | x_0) = Cat( x_0 \bar Q_t )

        Args:
            x0: (B, L) clean state ids
            t : (B,) timesteps in {1..T}

        Returns:
            probs_xt_given_x0: (B, L, K) where each row is a categorical distribution.
        """
        B, L = x0.shape
        x0_oh = to_one_hot(x0, self.num_states).to(self.device, self.dtype)  # (B,L,K)
        barQ_t = self.forward_product[t]                                     # (B,K,K)
        return torch.einsum("blk,bkj->blj", x0_oh, barQ_t)

    @torch.no_grad()
    def sample_forward(self, x0: torch.LongTensor, t: torch.LongTensor) -> torch.LongTensor:
        """
        Sample x_t ~ q(x_t | x_0) from the closed form marginal.

        This is often used in training (sample a random t, then sample x_t given x0).

        Args:
            x0: (B, L)
            t : (B,) in {1..T}

        Returns:
            xt: (B, L)
        """
        probs = self.forward_distribution(x0, t)
        return sample_categorical(probs)

    # ----------------------------
    # Posterior: q(x_{t-1} | x_t, x_0)
    # ----------------------------

    @torch.no_grad()
    def posterior_distribution(self, x_t: torch.LongTensor, x0: torch.LongTensor, t: torch.LongTensor) -> torch.Tensor:
        """
        Compute the exact forward posterior (paper Eq. 3):
            q(x_{t-1} | x_t, x_0)
            = Cat( ((x_t Q_t^T) ⊙ (x_0 \bar Q_{t-1})) / Z )

        Args:
            x_t: (B, L) state ids at time t
            x0 : (B, L) clean state ids
            t  : (B,) timesteps in {1..T}

        Returns:
            probs_x_tm1_given_xt_x0: (B, L, K)
        """
        if not (t >= 1).all():
            raise ValueError("t must be in {1..T}")

        x_t_oh = to_one_hot(x_t, self.num_states).to(self.device, self.dtype)  # (B,L,K)
        x0_oh  = to_one_hot(x0,  self.num_states).to(self.device, self.dtype)  # (B,L,K)

        Q_t = self.forward_transition[t - 1]         # (B,K,K) with Q_t at index t-1
        barQ_tm1 = self.forward_product[t - 1]       # (B,K,K) is \bar Q_{t-1}

        term_from_xt = torch.einsum("blk,bjk->blj", x_t_oh, Q_t.transpose(-1, -2))  # x_t Q_t^T
        term_from_x0 = torch.einsum("blk,bkj->blj", x0_oh, barQ_tm1)                # x0 \barQ_{t-1}

        unnormalized = term_from_xt * term_from_x0                                  # ⊙
        return unnormalized / unnormalized.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    # ----------------------------
    # Reverse: p_θ(x_{t-1} | x_t) from x0-pred (paper Eq. 4)
    # ----------------------------

    @torch.no_grad()
    def _reverse_distribution(
        self,
        x_t: torch.LongTensor,
        t: torch.LongTensor,
        p_x0_given_xt: torch.Tensor,
    ) -> torch.Tensor:
        """
        Construct p_θ(x_{t-1} | x_t) from an x0-prediction head, matching paper Eq. (4).

        Paper:
        p_θ(x_{t-1} | x_t) = Σ_{x̂0} q(x_{t-1} | x_t, x̂0) p_θ(x̂0 | x_t)

        For small K we can compute it without enumerating x̂0 explicitly:

        p_θ(x_{t-1} | x_t) ∝ (x_t Q_t^T) ⊙ (p_θ(x0|x_t) \bar Q_{t-1})

        Args:
            x_t: (B, L) state ids at time t
            t  : (B,) timesteps in {1..T}
            p_x0_given_xt: (B, L, K) probabilities representing p_θ(x0 | x_t, t)

        Returns:
            probs_x_tm1_given_xt: (B, L, K)
        """
        x_t_oh = to_one_hot(x_t, self.num_states).to(self.device, self.dtype)
        p_x0_given_xt = p_x0_given_xt.to(self.device, self.dtype)

        Q_t = self.forward_transition[t - 1]   # (B,K,K)
        barQ_tm1 = self.forward_product[t - 1] # (B,K,K)

        term_from_xt = torch.einsum("blk,bjk->blj", x_t_oh, Q_t.transpose(-1, -2))
        term_from_model = torch.einsum("blk,bkj->blj", p_x0_given_xt, barQ_tm1)

        unnormalized = term_from_xt * term_from_model
        return unnormalized / unnormalized.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    @torch.no_grad()
    def sample_reverse(
        self,
        x_t: torch.LongTensor,
        t: torch.LongTensor,
        p_x0_given_xt: torch.Tensor,
    ) -> torch.LongTensor:
        """
        Sample x_{t-1} ~ p_θ(x_{t-1} | x_t) using the x0-prediction parameterization.

        Args:
            x_t: (B, L)
            t  : (B,) in {1..T}
            p_x0_given_xt: (B, L, K)

        Returns:
            x_{t-1}: (B, L)
        """
        probs = self._reverse_distribution(x_t, t, p_x0_given_xt)
        return sample_categorical(probs)

