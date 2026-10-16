"""Accelerated decoding engine for the IndexTTS GPT (attn_backend="flash").

One sequence at a time (batch 1): paged fp16 KV cache, prefill with
flash_attn_varlen_func and every decode step captured in a CUDA graph. Samples like
transformers with num_beams=1 (repetition_penalty, temperature, top-k, top-p); no
beam search.
"""
import torch
from torch import nn

from .attention import reset_forward_context, set_forward_context
from .kv_manager import KVCacheManager, Seq


class AccelInferenceEngine:
    def __init__(
        self,
        model,
        lm_head,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        block_size: int = 256,
        num_blocks: int = 16,
        use_cuda_graph: bool = True,
    ):
        """
        Args:
            model: the GPT2 with accel attention (GPT2AccelModel).
            lm_head: final_norm + mel_head.
            block_size / num_blocks: geometry of the paged KV pool.
            use_cuda_graph: capture the decode step in a CUDA graph.
        """
        self.model = model
        self.lm_head = lm_head
        self.block_size = block_size
        self.use_cuda_graph = use_cuda_graph and torch.cuda.is_available()
        self.hidden_size = model.config.hidden_size
        self.kv_manager = KVCacheManager(num_layers, num_heads, head_dim, block_size, num_blocks)
        self.kv_manager.wire_kv_cache_to_model(model)
        self.graph = None
        self.graph_vars = None

    def _capture_cuda_graph(self, mel_embedding, mel_pos_embedding):
        # The block-table width is baked into the graph, and flash_attn_with_kvcache uses
        # it to decide how to split the work: changing it changes the output in the last
        # bit. It covers 2048 tokens; a longer sequence falls back to the non-graph path.
        max_num_blocks = (2048 + self.block_size - 1) // self.block_size
        dtype = next(self.model.parameters()).dtype
        gv = {
            "input_ids": torch.ones(1, dtype=torch.int64, device="cuda"),
            "positions": torch.ones(1, dtype=torch.int64, device="cuda"),
            "slot_mapping": torch.zeros(1, dtype=torch.int32, device="cuda"),
            "context_lens": torch.full((1,), 2, dtype=torch.int32, device="cuda"),
            "block_tables": torch.zeros(1, max_num_blocks, dtype=torch.int32, device="cuda"),
            "inputs_embeds": torch.zeros(1, self.hidden_size, dtype=dtype, device="cuda"),
            "outputs": torch.zeros(1, self.hidden_size, dtype=dtype, device="cuda"),
        }

        def step():
            emb = mel_embedding(gv["input_ids"])
            pos_emb = mel_pos_embedding.emb(torch.clamp(gv["positions"], min=0))
            gv["inputs_embeds"][:] = emb + pos_emb
            out = self.model(inputs_embeds=gv["inputs_embeds"].unsqueeze(1))
            gv["outputs"][:] = out.squeeze(1)

        set_forward_context(False, slot_mapping=gv["slot_mapping"], context_lens=gv["context_lens"],
                            block_tables=gv["block_tables"])
        step()  # warmup
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            step()
        torch.cuda.synchronize()
        reset_forward_context()
        self.graph, self.graph_vars = graph, gv

    def _decode(self, seq: Seq, mel_embedding, mel_pos_embedding):
        """One decode step for the last token of `seq`. Returns the hidden state [1, C]."""
        pos = len(seq) - seq.num_prompt_tokens  # mel position of the code
        slot = seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1
        gv = self.graph_vars
        if gv is not None and len(seq.block_table) <= gv["block_tables"].size(1):
            gv["input_ids"].fill_(seq.last_token)
            gv["positions"].fill_(pos)
            gv["slot_mapping"].fill_(slot)
            gv["context_lens"].fill_(len(seq))
            if len(seq.block_table) > self._graph_blocks:  # a new block was opened
                self._graph_blocks = len(seq.block_table)
                gv["block_tables"][0, self._graph_blocks - 1] = seq.block_table[-1]
            self.graph.replay()
            return gv["outputs"]

        set_forward_context(
            False,
            slot_mapping=torch.tensor([slot], dtype=torch.int32, device="cuda"),
            context_lens=torch.tensor([len(seq)], dtype=torch.int32, device="cuda"),
            block_tables=torch.tensor([seq.block_table], dtype=torch.int32, device="cuda"),
        )
        emb = mel_embedding(torch.tensor([seq.last_token], device="cuda"))
        emb = emb + mel_pos_embedding.emb(torch.tensor([pos], device="cuda"))
        out = self.model(inputs_embeds=emb.to(next(self.model.parameters()).dtype).unsqueeze(1))
        reset_forward_context()
        return out.squeeze(1)

    def _sample(self, logits, temperature, top_k, top_p, rp, seen):
        """One token, in transformers' order: repetition penalty on the raw logits,
        temperature, top-k, top-p and sampling (exponential race)."""
        if rp != 1.0:
            logits = torch.where(seen, torch.where(logits < 0, logits * rp, logits / rp), logits)
        if temperature <= 0:
            tok = torch.argmax(logits, dim=-1).tolist()[0]
        else:
            logits = logits.float() / temperature
            if top_k > 0:
                kth = torch.topk(logits, min(top_k, logits.size(-1))).values[..., -1:]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            if top_p < 1.0:
                sl, idx = torch.sort(logits, descending=False)
                remove = sl.softmax(-1).cumsum(-1) <= (1 - top_p)
                remove[..., -1:] = False
                logits = logits.masked_fill(remove.scatter(1, idx, remove), float("-inf"))
            probs = logits.softmax(-1)
            tok = (probs / torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(-1).tolist()[0]
        seen[0, tok] = True
        return tok

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        stop_tokens,
        tts_embeddings: torch.Tensor,
        tts_mel_embedding: nn.Module,
        tts_text_pos_embedding: nn.Module,
        top_k: int = 0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
    ) -> torch.Tensor:
        """Generate codes for one sequence.

        Args:
            input_ids: [1, n] filler ids ([1]*(n-1) + [start_mel]); the real prompt is
                `tts_embeddings` ([1, n-1, C]: [pad][cond][text]) plus the start_mel.
            max_new_tokens: cap on the number of generated codes.
            stop_tokens: codes that end the sequence (included in the output).
            tts_mel_embedding / tts_text_pos_embedding: code and position embeddings.

        Returns:
            [1, n + generated], like transformers' generate().
        """
        if self.use_cuda_graph and self.graph is None:
            self._capture_cuda_graph(tts_mel_embedding, tts_text_pos_embedding)

        n = input_ids.size(1)
        vocab = tts_mel_embedding.num_embeddings
        seen = torch.zeros(1, vocab, dtype=torch.bool, device=input_ids.device)
        seen[0, input_ids[0]] = True  # like transformers: the prompt's filler ids count too
        knobs = (top_k, top_p, repetition_penalty, seen)
        seq = Seq(n, last_token=-1, block_size=self.block_size)
        self.kv_manager.allocate(seq)
        if self.graph_vars is not None:
            self.graph_vars["block_tables"].fill_(-1)
            first = seq.block_table[: self.graph_vars["block_tables"].size(1)]
            self.graph_vars["block_tables"][0, : len(first)] = torch.tensor(first, dtype=torch.int32)
            self._graph_blocks = len(first)
        try:
            # Prefill: [pad][cond][text] + start_mel at mel position 0, as in training
            # (it used to go at n-1, the position of the last prompt token).
            start_emb = tts_mel_embedding(input_ids[:, -1:])
            start_pos = torch.zeros((1, 1), device="cuda", dtype=torch.long)
            start_emb = start_emb + tts_text_pos_embedding.emb(start_pos)
            full_embeddings = torch.cat([tts_embeddings, start_emb], dim=1)
            model_dtype = next(self.model.parameters()).dtype
            if full_embeddings.dtype != model_dtype:
                full_embeddings = full_embeddings.to(model_dtype)

            slots = [seq.block_table[t // self.block_size] * self.block_size + t % self.block_size
                     for t in range(n)]
            cu_seqlens = torch.tensor([0, n], dtype=torch.int32, device="cuda")
            set_forward_context(True, cu_seqlens, cu_seqlens, n, n,
                                torch.tensor(slots, dtype=torch.int32, device="cuda"))
            hidden = self.model(inputs_embeds=full_embeddings)[:, -1, :]
            reset_forward_context()
            lm_dtype = next(self.lm_head.parameters()).dtype
            if hidden.dtype != lm_dtype:
                hidden = hidden.to(lm_dtype)

            generated = []
            token = self._sample(self.lm_head(hidden), temperature, *knobs)
            for step in range(max_new_tokens):
                generated.append(token)
                # The stop token goes in the output, as in transformers: without it,
                # infer_v2_5 warned about a max_mel_tokens cutoff on every call.
                if token in stop_tokens or step == max_new_tokens - 1:
                    break
                seq.append_token(token)
                self.kv_manager.append_to_seq(seq)
                hidden = self._decode(seq, tts_mel_embedding, tts_text_pos_embedding)
                token = self._sample(self.lm_head(hidden), temperature, *knobs)
        finally:
            self.kv_manager.free(seq)
            reset_forward_context()

        return torch.cat([input_ids, torch.tensor([generated], dtype=input_ids.dtype, device=input_ids.device)], dim=1)
