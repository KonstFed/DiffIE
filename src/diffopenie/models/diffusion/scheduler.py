import torch
import torch.nn as nn


class LinearScheduler(nn.Module):
    """
    Linear noise schedule for diffusion models.
    """
    def __init__(self, num_steps: int, beta_start: float = 0.0001, beta_end: float = 0.02):
        super().__init__()
        self.num_steps = num_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        # Compute noise schedule parameters
        betas = torch.linspace(beta_start, beta_end, num_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        sqrt_alpha_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - alphas_cumprod)
        alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0]), alphas_cumprod[:-1]]
        )
        
        # Register buffers - these will automatically be moved to the correct device
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alpha_cumprod", sqrt_alpha_cumprod)
        self.register_buffer("sqrt_one_minus_alpha_cumprod", sqrt_one_minus_alpha_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # compute variance for reverse process
        posterior_variance = betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)

    def q_sample(self, x_0: torch.FloatTensor, t: torch.LongTensor, noise: torch.FloatTensor | None = None) -> torch.FloatTensor:
        if noise is None:
            noise = torch.randn_like(x_0)

        # this .view are for broadcasting into any dimension of x_0
        coef1 = self.sqrt_alpha_cumprod[t].view(-1, *([1] * (x_0.dim() - 1)))
        coef2 = self.sqrt_one_minus_alpha_cumprod[t].view(-1, *([1] * (x_0.dim() - 1)))
        sample = coef1 * x_0 + coef2 * noise
        return sample

    def q_posterior_mean(self, x_t: torch.FloatTensor, t: torch.LongTensor, x0_pred: torch.FloatTensor) -> torch.FloatTensor:
        shape = (-1, *([1] * (x_t.dim() - 1)))

        beta_t = self.betas[t].view(shape)
        alpha_t = self.alphas[t].view(shape)
        alpha_bar_t = self.alphas_cumprod[t].view(shape)
        alpha_bar_prev_t = self.alphas_cumprod_prev[t].view(shape)

        coef1 = torch.sqrt(alpha_bar_prev_t) * beta_t / (1 - alpha_bar_t)
        coef2 = torch.sqrt(alpha_t) * (1 - alpha_bar_prev_t) / (1 - alpha_bar_t)

        return coef1 * x0_pred + coef2 * x_t 

    def p_sample(self, model: nn.Module, x_t: torch.FloatTensor, t: torch.LongTensor, condition: torch.Tensor = None) -> torch.FloatTensor:
        """
        Sample from the reverse process p(x_{t-1} | x_t) using the provided model.

        Args:
            model (nn.Module): Denoising model predicting x0.
                The model should accept (at least) x_t, t, and (optionally) cond as its inputs.
                Example usage: model(x_t, t, cond) -> x0_pred
            x_t (torch.FloatTensor): Noisy sample at timestep t [batch_size, ...].
            t (torch.LongTensor): Current timestep(s) [batch_size].
            condition (torch.Tensor, optional): Conditioning information for the model, if used.

        Returns:
            torch.FloatTensor: Sample from p(x_{t-1} | x_t).
        """
        shape = (-1, *([1] * (x_t.dim() - 1)))
        x0_pred = model(x_t, t, condition)

        mean = self.q_posterior_mean(x_t, t, x0_pred)
        variance = self.posterior_variance[t].view(shape)

        std = torch.sqrt(torch.clamp(variance, min=1e-20))

        # If t=0: no noise added
        nonzero_mask = (t > 0).float().view(shape)

        noise = torch.randn_like(x_t)
        return mean + nonzero_mask * std * noise
