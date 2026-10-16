import torch

from indextts25.s2mel.modules.diffusion_transformer import DiT


class CFM(torch.nn.Module):
    """Conditional flow matching: Euler ODE over the DiT, inference only."""

    def __init__(self, args):
        super().__init__()
        self.in_channels = args.DiT.in_channels
        if args.dit_type != "DiT":
            raise NotImplementedError(f"Unknown diffusion type {args.dit_type}")
        self.estimator = DiT(args)

    @torch.inference_mode()
    def inference(self, mu, x_lens, prompt, style, f0, n_timesteps, temperature=1.0, inference_cfg_rate=0.5):
        """
        Args:
            mu: semantic condition for the prompt + the target, (B, T, 512).
            x_lens: number of output frames, (B,).
            prompt: reference mel, (B, 80, T_prompt).
            style: global style of the reference, (B, 192).
            f0: unused (None).
            n_timesteps: number of Euler steps.

        Returns:
            the generated mel, (B, 80, T), with the prompt part zeroed.
        """
        B, T = mu.size(0), mu.size(1)
        z = torch.randn([B, self.in_channels, T], device=mu.device) * temperature
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device)
        return self.solve_euler(z, x_lens, prompt, mu, style, f0, t_span, inference_cfg_rate)

    def solve_euler(self, x, x_lens, prompt, mu, style, f0, t_span, inference_cfg_rate=0.5):
        t, n = t_span[0], prompt.size(-1)
        prompt_x = torch.zeros_like(x)
        prompt_x[..., :n] = prompt
        x[..., :n] = 0
        if inference_cfg_rate > 0:
            # The unconditional branch (zeroed prompt, style and content) goes in the same
            # batch. It is constant across all steps, so it is built only once.
            prompt_x, style, mu = (torch.cat([v, torch.zeros_like(v)]) for v in (prompt_x, style, mu))
        for step in range(1, len(t_span)):
            dt = t_span[step] - t_span[step - 1]
            if inference_cfg_rate > 0:
                dphi_dt, cfg_dphi_dt = self.estimator(
                    torch.cat([x, x]), prompt_x, x_lens, torch.stack([t, t]), style, mu).chunk(2)
                dphi_dt = (1.0 + inference_cfg_rate) * dphi_dt - inference_cfg_rate * cfg_dphi_dt
            else:
                dphi_dt = self.estimator(x, prompt_x, x_lens, t.unsqueeze(0), style, mu)
            x = x + dt * dphi_dt
            t = t + dt
            x[:, :, :n] = 0
        return x
