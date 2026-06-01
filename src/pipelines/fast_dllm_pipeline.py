from typing import Dict

import torch

from src.pipelines.diffusion_pipeline import DiffusionPipeline


class FastDLLMPipeline(DiffusionPipeline):
    @torch.no_grad()
    def generate(self, batch: Dict, n_return_sequences: int = 1, mode: str = "fast_dllm") -> torch.Tensor:
        """Fast DLLM-style blockwise decoding with optional KV cache.

        The implementation keeps behavioural parity with the previous version
        while improving readability and typing.
        """
        encoder_hidden = self.model.encode(batch)

        if n_return_sequences > 1:
            encoder_hidden = torch.repeat_interleave(encoder_hidden, n_return_sequences, dim=0)
            for key in batch.keys():
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = torch.repeat_interleave(batch[key], n_return_sequences, dim=0)

        batch_size = encoder_hidden.size(0)
        device = encoder_hidden.device

        cfg = self.config.get("fast_dllm", {})
        L = int(self.config.get("answer_length", 128))
        B = int(cfg.get("block_size", 32))
        K = (L + B - 1) // B
        T = int(cfg.get("steps_per_block", 10))
        strategy = cfg.get("strategy", "threshold")
        tau = float(cfg.get("threshold", 0.9))
        factor = float(cfg.get("factor", 0.5))
        use_cache = bool(cfg.get("use_cache", True))
        use_dual_cache = bool(cfg.get("use_dual_cache", False))

        mask_token_id = self.tokenizer.mask_token_id

        prompt = batch.get("prompt_ids", torch.empty((batch_size, 0), dtype=torch.long, device=device))
        prompt_len = prompt.size(1)

        masks = torch.full((batch_size, L), mask_token_id, dtype=torch.long, device=device)
        x = torch.cat([prompt, masks], dim=1)

        past_key_values = None

        for block_idx in range(1, K + 1):
            s = prompt_len + (block_idx - 1) * B
            e = min(prompt_len + block_idx * B, x.size(1))

            for step in range(1, T + 1):
                if use_cache:
                    model_input = x[:, :e] if not use_dual_cache else x
                    outputs = self.model.decode(
                        model_input,
                        encoder_hidden_states=encoder_hidden,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                    logits = outputs.logits
                    past_key_values = getattr(outputs, "past_key_values", None)
                    block_logits = logits[:, s:e, :]
                else:
                    outputs = self.model.decode(x, encoder_hidden_states=encoder_hidden)
                    block_logits = outputs.logits[:, s:e, :]

                block_x = x[:, s:e]
                is_masked = (block_x == mask_token_id)

                probs = torch.softmax(block_logits, dim=-1)
                confidences, preds = torch.max(probs, dim=-1)

                for b in range(batch_size):
                    masked_indices = is_masked[b].nonzero(as_tuple=True)[0]
                    if masked_indices.numel() == 0:
                        continue

                    b_conf = confidences[b, masked_indices]
                    b_preds = preds[b, masked_indices]

                    unmask_mask = torch.zeros_like(b_conf, dtype=torch.bool)

                    if strategy == "threshold":
                        unmask_mask = b_conf >= tau
                        if not unmask_mask.any():
                            unmask_mask[torch.argmax(b_conf)] = True

                    elif strategy == "factor":
                        sorted_conf, sorted_idx = torch.sort(b_conf, descending=True)
                        n = 0
                        for i in range(len(sorted_conf)):
                            if (i + 2) * (1 - sorted_conf[i].item()) < factor:
                                n = i + 1
                            else:
                                break
                        n = max(n, 1)
                        unmask_mask[sorted_idx[:n]] = True

                    tokens_to_unmask = masked_indices[unmask_mask]
                    x[b, s + tokens_to_unmask] = b_preds[unmask_mask]

                if not (x[:, s:e] == mask_token_id).any():
                    break

            # KV cache is updated via `past_key_values` when supported by the model

        return x[:, prompt_len:]