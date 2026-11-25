import inspect

import torch


class DDPMScheduler:
    def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=0.02,
                 device="cpu", schedule: str = "linear", prediction_type: str = "epsilon"):
        self.num_timesteps = int(num_timesteps)
        self.device = torch.device(device)
        # "epsilon": denoiser predicts the noise (classic DDPM).
        # "x0": denoiser predicts the clean target; sampling converts x0_hat to
        # eps_hat before the DDIM update. x0-parameterisation is far more stable
        # for small denoisers, whose low-noise epsilon predictions are poor.
        self.prediction_type = str(prediction_type or "epsilon").strip().lower()
        if self.prediction_type not in {"epsilon", "x0"}:
            raise ValueError(f"prediction_type must be 'epsilon' or 'x0', got {prediction_type!r}")

        if schedule == "cosine":
            self.betas = self._cosine_beta_schedule(num_timesteps).to(self.device)
        else:
            self.betas = torch.linspace(beta_start, beta_end, self.num_timesteps, device=self.device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0], device=self.device), self.alphas_cumprod[:-1]],
            dim=0,
        )

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        ).clamp(min=1e-20)

    @staticmethod
    def _cosine_beta_schedule(timesteps: int, s: float = 0.008):
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos((x / timesteps + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
        return torch.clip(betas, 1e-5, 0.999)

    def _to_device(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(self.device)

    def _ensure_BCL(self, condition: torch.Tensor, data_format: str = "BCL") -> torch.Tensor:
        if condition.dim() != 3:
            raise RuntimeError(f"condition should be 3D, got {tuple(condition.shape)}")

        data_format = str(data_format or "BCL").strip().upper()
        if data_format == "BCL":
            return condition
        if data_format == "BLC":
            return condition.permute(0, 2, 1).contiguous()
        raise ValueError(f"data_format must be 'BCL' or 'BLC', got {data_format!r}")

    @staticmethod
    def _filter_forward_kwargs(model, kwargs: dict) -> dict:
        forward = getattr(model, "forward", model)
        sig = inspect.signature(forward)
        params = sig.parameters
        accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
        if accepts_kwargs:
            return kwargs
        return {k: v for k, v in kwargs.items() if k in params}

    def add_noise(self, original_samples: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        original_samples = self._to_device(original_samples)
        noise = self._to_device(noise)
        t = t.to(self.device).long()

        s1 = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        s2 = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        return s1 * original_samples + s2 * noise

    def step(
        self,
        model_output: torch.Tensor,
        t: torch.Tensor,
        sample: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if t.dim() == 0:
            t0 = int(t.item())
        else:
            t0 = int(t[0].item())

        beta_t = self.betas[t0].view(1, 1, 1)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t0].view(1, 1, 1)
        sqrt_recip_alpha_t = self.sqrt_recip_alphas[t0].view(1, 1, 1)

        model_mean = sqrt_recip_alpha_t * (
            sample - beta_t * model_output / sqrt_one_minus_alpha_cumprod_t
        )

        if t0 > 0:
            noise = torch.randn(sample.shape, device=sample.device, dtype=sample.dtype, generator=generator)
            posterior_var_t = self.posterior_variance[t0].view(1, 1, 1)
            return model_mean + torch.sqrt(posterior_var_t) * noise
        return model_mean

    def _make_sampling_schedule(self, num_inference_steps: int, t_start: int | None = None) -> list[int]:
        num_inference_steps = int(max(1, num_inference_steps))
        t_max = self.num_timesteps - 1
        t_begin = int(t_start) if t_start is not None else t_max
        t_begin = max(0, min(t_begin, t_max))
        if num_inference_steps >= t_begin + 1:
            return list(range(t_begin, -1, -1))

        raw = torch.linspace(
            float(t_begin),
            0,
            steps=num_inference_steps,
            device=self.device,
        ).round().long().tolist()

        timesteps: list[int] = []
        for t in raw:
            t = int(t)
            if len(timesteps) == 0 or timesteps[-1] != t:
                timesteps.append(t)
        if timesteps[-1] != 0:
            timesteps.append(0)
        return timesteps

    def _ddim_step(
        self,
        model_output: torch.Tensor,
        sample: torch.Tensor,
        t_cur: int,
        t_prev: int,
        generator: torch.Generator | None = None,
        eta: float = 0.0,
    ) -> torch.Tensor:
        alpha_bar_t = self.alphas_cumprod[t_cur].view(1, 1, 1)
        if t_prev >= 0:
            alpha_bar_prev = self.alphas_cumprod[t_prev].view(1, 1, 1)
        else:
            alpha_bar_prev = torch.ones_like(alpha_bar_t)

        sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t.clamp(min=1e-20))
        sqrt_one_minus_alpha_bar_t = torch.sqrt((1.0 - alpha_bar_t).clamp(min=1e-20))
        pred_x0 = (sample - sqrt_one_minus_alpha_bar_t * model_output) / sqrt_alpha_bar_t

        eta = float(max(0.0, eta))
        sigma_t = torch.zeros_like(alpha_bar_t)
        if eta > 0.0 and t_prev >= 0:
            ratio = ((1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t).clamp(min=1e-20)).clamp(min=0.0)
            variance = (1.0 - alpha_bar_t / alpha_bar_prev.clamp(min=1e-20)).clamp(min=0.0)
            sigma_t = eta * torch.sqrt(ratio * variance)

        pred_dir = torch.sqrt((1.0 - alpha_bar_prev - sigma_t.pow(2)).clamp(min=0.0)) * model_output
        x_prev = torch.sqrt(alpha_bar_prev.clamp(min=1e-20)) * pred_x0 + pred_dir

        if eta > 0.0 and t_prev >= 0:
            noise = torch.randn(sample.shape, device=sample.device, dtype=sample.dtype, generator=generator)
            x_prev = x_prev + sigma_t * noise

        return x_prev

    @torch.no_grad()
    def sample(
        self,
        model,
        condition: torch.Tensor,
        seq_len: int | None = None,
        strict_seq_len: bool = True,
        debug: bool = False,
        condition_format: str = "BCL",
        known_future_format: str | None = None,
        coarse_label: torch.Tensor | None = None,
        known_future: torch.Tensor | None = None,
        seed: int | None = None,
        sampler: str = "ddpm",
        num_inference_steps: int | None = None,
        eta: float = 0.0,
        t_start: int | None = None,
    ) -> torch.Tensor:
        condition = self._to_device(condition)
        condition = self._ensure_BCL(condition, data_format=condition_format)

        B, C, condition_len = condition.shape
        if seq_len is None:
            seq_len = condition_len
        seq_len = int(seq_len)

        if debug:
            print(f"[Scheduler] condition shape={tuple(condition.shape)}, seq_len={seq_len}, strict={strict_seq_len}")

        sampler = str(sampler).strip().lower()
        if sampler not in {"ddpm", "ddim"}:
            raise ValueError(f"Unknown sampler: {sampler}")

        if num_inference_steps is None:
            num_inference_steps = self.num_timesteps
        num_inference_steps = int(max(1, num_inference_steps))

        if sampler == "ddpm" and num_inference_steps < self.num_timesteps:
            raise ValueError("DDPM sampler requires full diffusion steps; use DDIM for fast sampling.")

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(int(seed))

        timesteps = self._make_sampling_schedule(num_inference_steps, t_start=t_start)
        t_first = timesteps[0]
        x = torch.randn(B, C, seq_len, device=self.device, generator=generator)
        if t_first < self.num_timesteps - 1:
            alpha_first = self.alphas_cumprod[t_first]
            x = torch.sqrt(1.0 - alpha_first) * x

        label = None
        if coarse_label is not None:
            label = coarse_label.to(self.device).long()
            if label.dim() == 0:
                label = label.view(1)
        known = None
        if known_future is not None:
            known = self._to_device(known_future)
            known = self._ensure_BCL(known, data_format=known_future_format or condition_format)

        for step_idx, i in enumerate(timesteps):
            t = torch.full((B,), int(i), device=self.device, dtype=torch.long)

            kwargs = self._filter_forward_kwargs(
                model,
                {"coarse_label": label, "known_future": known},
            )
            pred_noise = model(x, t, condition, **kwargs)

            if self.prediction_type == "x0":
                alpha_bar_t = self.alphas_cumprod[int(i)].view(1, 1, 1)
                x0_hat = pred_noise
                pred_noise = (x - torch.sqrt(alpha_bar_t) * x0_hat) / torch.sqrt(
                    (1.0 - alpha_bar_t).clamp(min=1e-12)
                )

            if pred_noise.dim() != 3:
                raise RuntimeError(
                    f"[Scheduler] model output should be [B,C,L], got {tuple(pred_noise.shape)}"
                )

            if pred_noise.shape != x.shape:
                if strict_seq_len:
                    raise RuntimeError(
                        f"[Scheduler] model output shape {tuple(pred_noise.shape)} != sample shape {tuple(x.shape)}"
                    )

                batch_pred, channels_pred, length_pred = pred_noise.shape
                if channels_pred > C:
                    pred_noise = pred_noise[:, :C, :]
                elif channels_pred < C:
                    pred_noise = torch.cat(
                        [pred_noise, pred_noise.new_zeros((batch_pred, C - channels_pred, length_pred))],
                        dim=1,
                    )

                if length_pred > seq_len:
                    pred_noise = pred_noise[:, :, :seq_len]
                elif length_pred < seq_len:
                    pred_noise = torch.cat(
                        [pred_noise, pred_noise.new_zeros((batch_pred, C, seq_len - length_pred))],
                        dim=2,
                    )

            if sampler == "ddim":
                prev_i = timesteps[step_idx + 1] if (step_idx + 1) < len(timesteps) else -1
                x = self._ddim_step(
                    pred_noise,
                    x,
                    t_cur=int(i),
                    t_prev=int(prev_i),
                    generator=generator,
                    eta=eta,
                )
            else:
                x = self.step(pred_noise, t, x, generator=generator)

        return x
