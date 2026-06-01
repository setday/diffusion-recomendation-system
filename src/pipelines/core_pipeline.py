import math
from typing import Dict

import torch

from src.pipelines.diffusion_pipeline import DiffusionPipeline


class CorePipeline(DiffusionPipeline):
    @torch.no_grad()
    def generate(self, batch: Dict, n_return_sequences: int = 1, mode: str = "core") -> torch.Tensor:
        """Generate sequences using the CoRe remasking strategy.

        This method is an inference-only routine and is intentionally non-differentiable.
        """
        encoder_hidden = self.model.encode(batch)

        if n_return_sequences > 1:
            encoder_hidden = torch.repeat_interleave(
                encoder_hidden, n_return_sequences, dim=0
            )
            for key in batch.keys():
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = torch.repeat_interleave(batch[key], n_return_sequences, dim=0)

        batch_size = encoder_hidden.size(0)
        device = encoder_hidden.device

        core_cfg = self.config.get("core_remasking", {})
        N = int(core_cfg.get("diffusion_steps", 128))
        gamma_s = float(core_cfg.get("gamma_s", 0.25))
        gamma_e = float(core_cfg.get("gamma_e", 0.75))
        E = int(core_cfg.get("revision_interval", 8))
        candidate_size = int(core_cfg.get("candidate_size", 32))
        remasking_limit = int(core_cfg.get("remasking_limit", 1))
        L = int(self.config.get("answer_length", 128))

        mask_token_id = self.tokenizer.mask_token_id

        prompt = batch.get("prompt_ids", torch.empty((batch_size, 0), dtype=torch.long, device=device))
        prompt_len = prompt.size(1)
        masks = torch.full((batch_size, L), mask_token_id, dtype=torch.long, device=device)
        y = torch.cat([prompt, masks], dim=1)

        # k_t is the number of tokens to unmask per diffusion step
        k_t = max(1, math.ceil(L / N))

        for t in range(1, N + 1):
            outputs = self.model.decode(y, encoder_hidden_states=encoder_hidden)
            probs = torch.softmax(outputs.logits, dim=-1)

            in_window = (gamma_s <= t / N < gamma_e)
            is_revision_step = (t % E == 0)

            # Revision step with remasking
            if in_window and is_revision_step and remasking_limit > 0:
                is_unmasked = (y[:, prompt_len:] != mask_token_id)

                top2_vals, _ = torch.topk(probs[:, prompt_len:, :], 2, dim=-1)
                margins = top2_vals[:, :, 0] - top2_vals[:, :, 1]
                margins[~is_unmasked] = float("inf")

                s_indices = []
                for b in range(batch_size):
                    num_unmasked = int(is_unmasked[b].sum().item())
                    actual_m = min(candidate_size, num_unmasked)
                    if actual_m > 0:
                        _, min_margin_idx = torch.topk(margins[b], actual_m, largest=False)
                        s_indices.append(min_margin_idx + prompt_len)
                    else:
                        s_indices.append(torch.empty(0, dtype=torch.long, device=device))

                y_tilde = y.clone()
                for b, idxs in enumerate(s_indices):
                    if idxs.numel() > 0:
                        y_tilde[b, idxs] = mask_token_id

                outputs_tilde = self.model.decode(y_tilde, encoder_hidden_states=encoder_hidden)
                probs_tilde = torch.softmax(outputs_tilde.logits, dim=-1)

                for b, idxs in enumerate(s_indices):
                    if idxs.numel() == 0:
                        continue

                    original = y[b, idxs]
                    prob_i = probs_tilde[b, idxs].gather(1, original.unsqueeze(1)).squeeze(1)
                    instability = -torch.log(prob_i + 1e-9)

                    actual_k_rm = min(remasking_limit, idxs.numel())
                    _, max_inst_idx = torch.topk(instability, actual_k_rm)
                    chosen_positions = idxs[max_inst_idx]

                    new_tokens = torch.argmax(probs_tilde[b, chosen_positions], dim=-1)
                    y[b, chosen_positions] = new_tokens

            # Greedy unmasking step for all batches
            for b in range(batch_size):
                masked_mask = (y[b, prompt_len:] == mask_token_id)
                if not masked_mask.any():
                    continue

                mask_probs, mask_preds = torch.max(probs[b, prompt_len:], dim=-1)
                mask_probs[~masked_mask] = -1.0

                actual_k = min(k_t, int(masked_mask.sum().item()))
                _, topk_idx = torch.topk(mask_probs, actual_k)

                y[b, prompt_len + topk_idx] = mask_preds[topk_idx]

        return y[:, prompt_len:]
