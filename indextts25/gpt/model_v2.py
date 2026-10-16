"""IndexTTS-2.5 GPT (UnifiedVoice): text + conditioning -> semantic codes.

2.5 mode only: the speaker is the 192-float CAMPPlus vector (`spk_emb_proj`) and the
emotion comes from a Conformer + Perceiver over the w2v-bert features. Generates with
the transformers GPT2 (attn_backend="sdpa") or the accel engine (attn_backend="flash").
"""
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GenerationMixin, GPT2Config, GPT2Model, GPT2PreTrainedModel, LogitsProcessorList
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from indextts25.gpt.conformer_encoder import ConformerEncoder
from indextts25.gpt.perceiver import PerceiverResampler
from indextts25.utils.tokenizer import LANGUAGE_DICT
from indextts25.utils.typical_sampling import TypicalLogitsWarper


def null_position_embeddings(range, dim):
    # A scalar zero: adding it changes nothing and, unlike an fp32 tensor the size of
    # the input, it does not promote the embeddings to fp32 when the GPT runs in bf16.
    return torch.zeros((), device=range.device)


def _past_len(past):
    """Already-cached length, for both the transformers >=5 Cache and the nested tuple
    of 4.x. Do NOT use `if past:`: an empty Cache is truthy, so on the first step it
    would trim input_ids and drop the conditioning prefix."""
    if past is None:
        return 0
    if hasattr(past, "get_seq_length"):
        try:
            return int(past.get_seq_length() or 0)
        except TypeError:
            return 0
    if len(past) == 0 or past[0] is None or past[0][0] is None:
        return 0
    return int(past[0][0].shape[2])


# Explicit GenerationMixin: transformers 5.x removed it from PreTrainedModel.
class GPT2InferenceModel(GPT2PreTrainedModel, GenerationMixin):
    """Adapts the GPT2 to `generate()`: the [pad][cond][text] prefix goes in as cached
    embeddings (`store_mel_emb`) behind fake input_ids, and each generated code is
    embedded with `embeddings` + its mel position."""

    def __init__(self, config, gpt, text_pos_emb, embeddings, norm, linear):
        super().__init__(config)
        # Despite the name, `text_pos_emb` is the mel position embedding.
        self.transformer = gpt
        self.text_pos_embedding = text_pos_emb
        self.embeddings = embeddings
        self.final_norm = norm
        self.lm_head = nn.Sequential(norm, linear)
        self.cached_mel_emb = None

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def store_mel_emb(self, mel_emb):
        self.cached_mel_emb = mel_emb

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)  # usually None
        # only last token for inputs_ids if past is defined in kwargs
        past_len = _past_len(past_key_values)
        if past_len > 0:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 0)
            if past_len > 0:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

    def forward(
            self,
            input_ids=None,
            past_key_values=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        assert self.cached_mel_emb is not None
        assert inputs_embeds is None  # Not supported by this inference model.
        assert labels is None  # Training not supported by this inference model.
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        # Create embedding
        mel_len = self.cached_mel_emb.shape[1]
        if input_ids.shape[1] != 1:
            text_inputs = input_ids[:, mel_len:]
            text_emb = self.embeddings(text_inputs)
            text_emb = text_emb + self.text_pos_embedding(text_emb)
            if self.cached_mel_emb.shape[0] != text_emb.shape[0]:
                mel_emb = self.cached_mel_emb.repeat_interleave(
                    text_emb.shape[0] // self.cached_mel_emb.shape[0], 0
                )
            else:  # this outcome only occurs once per loop in most cases
                mel_emb = self.cached_mel_emb
            emb = torch.cat([mel_emb, text_emb], dim=1)
        else:
            emb = self.embeddings(input_ids)
            # Code k goes at mel position k (start_mel at 0), as in training: upstream
            # shifted it by one position and text conditioning got worse (+0.5 nats/token
            # of NLL on real audio). The position comes from the cache length, not from
            # attention_mask, which transformers 5.x omits when it is all ones.
            emb = emb + self.text_pos_embedding.get_fixed_embedding(
                _past_len(past_key_values) - mel_len, emb.device
            )
        transformer_outputs = self.transformer(
            inputs_embeds=emb,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        lm_logits = self.lm_head(hidden_states)

        if not return_dict:
            return (lm_logits,) + transformer_outputs[1:]

        return CausalLMOutputWithCrossAttentions(
            loss=None,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

    @staticmethod
    def _reorder_cache(past, beam_idx):
        """Reorders the KV cache across beams (beam search / beam sample)."""
        # transformers >=5 passes a Cache object that reorders itself; 4.x passed a
        # nested tuple of tensors (which may now contain None entries).
        if hasattr(past, "reorder_cache"):
            past.reorder_cache(beam_idx)
            return past
        return tuple(
            tuple(
                None if past_state is None
                else past_state.index_select(0, beam_idx.to(past_state.device))
                for past_state in layer_past
            )
            for layer_past in past
        )


class LearnedPositionEmbeddings(nn.Module):
    def __init__(self, seq_len, model_dim, init=.02):
        super().__init__()
        self.emb = nn.Embedding(seq_len, model_dim)
        # Initializing this way is standard for GPT-2
        self.emb.weight.data.normal_(mean=0.0, std=init)

    def forward(self, x):
        sl = x.shape[1]
        return self.emb(torch.arange(0, sl, device=x.device))

    def get_fixed_embedding(self, ind, dev):
        return self.emb(torch.tensor([ind], device=dev)).unsqueeze(0)


def build_hf_gpt_transformer(layers, model_dim, heads, max_mel_seq_len, max_text_seq_len):
    """transformers GPT-2 without its own embeddings: positions come from the mel and
    text LearnedPositionEmbeddings."""
    gpt_config = GPT2Config(vocab_size=256,  # Unused.
                            n_positions=max_mel_seq_len + max_text_seq_len,
                            n_ctx=max_mel_seq_len + max_text_seq_len,
                            n_embd=model_dim,
                            n_layer=layers,
                            n_head=heads,
                            activation_function="gelu_pytorch_tanh",
                            use_cache=False,
                            # The defaults (50256, from GPT-2's vocab) fall outside this
                            # vocab and transformers 5.x warns about it on stderr.
                            bos_token_id=None, eos_token_id=None)
    gpt = GPT2Model(gpt_config)
    # Override the built in positional embeddings
    del gpt.wpe
    gpt.wpe = functools.partial(null_position_embeddings, dim=model_dim)
    # Built-in token embeddings are unused.
    del gpt.wte
    # transformers 4.x registers a per-layer n_pos² causal mask (~134 MB in total)
    # that only eager attention uses; this GPT runs on SDPA. 5.x no longer creates it.
    for block in gpt.h:
        block.attn._buffers.pop("bias", None)
    return gpt, LearnedPositionEmbeddings(max_mel_seq_len, model_dim), LearnedPositionEmbeddings(max_text_seq_len, model_dim)


class UnifiedVoice(nn.Module):
    def __init__(self, layers=8, model_dim=512, heads=8, max_text_tokens=120, max_mel_tokens=250,
                 max_conditioning_inputs=1, number_text_tokens=256, start_text_token=0, stop_text_token=1,
                 number_mel_codes=8194, start_mel_token=8192, stop_mel_token=8193, emo_condition_module=None,
                 use_accel=False, **_):
        """
        Takes the `gpt:` section of config.yaml; the IndexTTS-2.0 conditioning keys
        (condition_type, condition_module, ...) are ignored.

        Args:
            layers: number of transformer layers.
            model_dim: model dimension.
            heads: attention heads (model_dim must be divisible by it).
            max_text_tokens / max_mel_tokens: max number of text tokens and of codes.
            emo_condition_module: config of the emotion Conformer.
            use_accel: build the accel engine (flash-attn) in post_init_gpt2_config.
        """
        super().__init__()
        self.number_text_tokens = number_text_tokens
        self.start_text_token = start_text_token
        self.stop_text_token = stop_text_token
        self.number_mel_codes = number_mel_codes
        self.start_mel_token = start_mel_token
        self.stop_mel_token = stop_mel_token
        self.layers = layers
        self.heads = heads
        self.max_mel_tokens = max_mel_tokens
        self.max_text_tokens = max_text_tokens
        self.model_dim = model_dim
        self.max_conditioning_inputs = max_conditioning_inputs
        self.emo_cond_mask_pad = nn.ConstantPad1d((1, 0), True)
        self.spk_emb_proj = nn.Linear(192, model_dim)

        self.emo_conditioning_encoder = ConformerEncoder(input_size=1024,
                                                         output_size=emo_condition_module['output_size'],
                                                         linear_units=emo_condition_module['linear_units'],
                                                         attention_heads=emo_condition_module['attention_heads'],
                                                         num_blocks=emo_condition_module['num_blocks'],
                                                         input_layer=emo_condition_module['input_layer'])
        self.emo_perceiver_encoder = PerceiverResampler(1024, dim_context=emo_condition_module['output_size'],
                                                        ff_mult=emo_condition_module['perceiver_mult'],
                                                        heads=emo_condition_module['attention_heads'],
                                                        num_latents=1)

        self.text_embedding = nn.Embedding(self.number_text_tokens + 1, model_dim)
        self.lang_embedding = nn.Embedding(len(LANGUAGE_DICT) + 1, model_dim)
        self.emo_layer = nn.Linear(model_dim, model_dim)
        self.emovec_layer = nn.Linear(1024, model_dim)
        self.mel_embedding = nn.Embedding(self.number_mel_codes, model_dim)
        self.gpt, self.mel_pos_embedding, self.text_pos_embedding = build_hf_gpt_transformer(
            layers, model_dim, heads, self.max_mel_tokens + 2 + self.max_conditioning_inputs,
            self.max_text_tokens + 2)
        self.final_norm = nn.LayerNorm(model_dim)
        # The checkpoint also carries text_head (77 M parameters, ~300 MB), used only by
        # the training forward: it is not created and its key is ignored on load.
        self.mel_head = nn.Linear(model_dim, self.number_mel_codes)

        self.use_accel = use_accel
        self.accel_engine = None  # built in post_init_gpt2_config

    def post_init_gpt2_config(self):
        seq_length = self.max_mel_tokens + self.max_text_tokens + 2
        gpt_config = GPT2Config(
            vocab_size=self.number_mel_codes,
            n_positions=seq_length,
            n_ctx=seq_length,
            n_embd=self.model_dim,
            n_layer=self.layers,
            n_head=self.heads,
            activation_function="gelu_pytorch_tanh",
            use_cache=True,
            # generate() gets the mel tokens explicitly; 50256 only triggered a warning.
            bos_token_id=None, eos_token_id=None,
        )

        if self.use_accel and self.mel_head.weight.is_cuda:
            from indextts25.accel import AccelInferenceEngine, GPT2AccelModel

            device = self.mel_head.weight.device
            # Built on "meta" (no memory and no random init, which cost ~4.5 s on CPU) and
            # assign=True shares the tensors with self.gpt, in their dtype, instead of
            # copying them: in fp32 that was ~1.9 GB of duplicated VRAM, and with half=True
            # the fp16 copy also forced the bf16 autocast to recast every weight on each
            # CUDA graph step. The attention hands q/k/v to flash-attn in fp16 anyway.
            # wte/wpe are unused: the codes are embedded outside.
            with torch.device("meta"):
                accel_gpt = GPT2AccelModel(gpt_config)
            accel_gpt.load_state_dict(self.gpt.state_dict(), strict=False, assign=True)
            del accel_gpt.wte, accel_gpt.wpe
            accel_gpt = accel_gpt.to(device).eval()

            with torch.cuda.device(device):
                self.accel_engine = AccelInferenceEngine(
                    model=accel_gpt,
                    lm_head=nn.Sequential(self.final_norm, self.mel_head),
                    num_layers=self.layers,
                    num_heads=self.heads,
                    head_dim=self.model_dim // self.heads,
                    block_size=256,
                    # Just enough for the longest possible sequence (prompt + max_mel_tokens):
                    # 10 blocks instead of 16, ~190 MB less KV in fp16.
                    num_blocks=-(-(self.max_text_tokens + self.max_mel_tokens + 8) // 256),
                    use_cuda_graph=True,
                )
        self.inference_model = GPT2InferenceModel(
            gpt_config,
            self.gpt,
            self.mel_pos_embedding,
            self.mel_embedding,
            self.final_norm,
            self.mel_head,
        ).eval()
        self.gpt.wte = self.mel_embedding

    def prepare_gpt_inputs(
        self,
        conditional_latents: torch.Tensor,
        text_inputs: torch.Tensor,
        langs: torch.Tensor = None,
    ):
        """
        Prepare the inputs for the GPT2InferenceModel to generate.
        Args:
            conds_latent: (b, 32, dim) audio conditioning embedding by `get_conditioning()`
            text_inputs: (b, L)
        Returns:
            input_ids: (b, s+1) the input ids for the GPT2InferenceModel.generate()
            inputs_embeds: (b, s+1, dim) the input embeddings for the GPT2InferenceModel.forward()
            attention_mask: (b, s+1) the attention mask for the GPT2InferenceModel.generate()
        """
        b, L = text_inputs.shape[:2]
        device = text_inputs.device
        single_cond = conditional_latents.ndim == 3 and conditional_latents.shape[0] == 1
        if not single_cond:
            assert conditional_latents.shape[0] == b, f"batch size mismatch: {conditional_latents.shape[0]} vs {b}"
        batched_mel_emb = []
        attention_masks = []
        target_len = conditional_latents.shape[1] + L + 2
        for i in range(b):
            # Only padding stops are removed. The start (id 0) is also "!" in this
            # tokenizer: filtering it out deleted the exclamation marks from the text.
            valid_mask = text_inputs[i] != self.stop_text_token
            text_input = text_inputs[i][valid_mask]
            text_input = F.pad(text_input, (1, 0), value=self.start_text_token)
            text_input = F.pad(text_input, (0, 1), value=self.stop_text_token)
            text_input_pos = torch.arange(0, text_input.size(-1), device=device)
            text_emb = self.text_embedding(text_input) + self.text_pos_embedding.emb(text_input_pos)
            if langs is not None:
                text_emb += self.lang_embedding(langs[i])
            # concatenate [conditional latents][text embeddings]
            conds_text_emb = [
                conditional_latents.squeeze(0) if single_cond else conditional_latents[i],
                text_emb,
            ]
            # +1 for the start_mel_token
            attention_mask = torch.ones(target_len+1, dtype=torch.long, device=device)
            # check this text input is padded
            padding: int = L + 2 - text_input.size(-1)
            # pad left of [cond][text] -> [pad][cond][text]
            if padding > 0:
                pad = torch.zeros((padding, conditional_latents.size(-1)), dtype=text_emb.dtype, device=device) # [p, dim]
                conds_text_emb.insert(0, pad)
                attention_mask[:padding] = 0
            mel_emb = torch.cat(conds_text_emb) #[s, dim]
            assert mel_emb.shape[0] == target_len, f"mel_emb.shape: {mel_emb.shape}, target_len: {target_len}"
            batched_mel_emb.append(mel_emb)
            attention_masks.append(attention_mask)
        # [b, s, dim]
        batched_mel_emb = torch.stack(batched_mel_emb, dim=0)
        # [b, s+1]
        attention_mask = torch.stack(attention_masks, dim=0)
        # [b, s+1]
        fake_inputs = torch.ones(
            (
                batched_mel_emb.shape[0],
                batched_mel_emb.shape[1] + 1,  # +1 for the start_mel_token
            ),
            dtype=torch.long,
            device=device,
        )
        fake_inputs[:, -1] = self.start_mel_token
        return fake_inputs, batched_mel_emb, attention_mask

    def inference_speech(self, text_inputs, langs, emo_vec, campplus_embedding, max_generate_length=None,
                         typical_sampling=False, typical_mass=.9, **hf_generate_kwargs):
        """
        Args:
            text_inputs: (b, L) text ids.
            langs: (b,) language id.
            emo_vec: (b, dim) already-merged emotion vector (merge_emovec).
            campplus_embedding: (b, 192) CAMPPlus speaker style.
            max_generate_length: cap on the number of codes to generate.
            hf_generate_kwargs: for `GPT2InferenceModel.generate()`. The accel engine uses
                temperature, top_k, top_p and repetition_penalty, without beams.
        Returns:
            the generated codes, (b, n).
        """
        speech_conditioning_latent = self.spk_emb_proj(campplus_embedding)
        if speech_conditioning_latent.ndim != 3:
            speech_conditioning_latent = speech_conditioning_latent.unsqueeze(0)
        conds_latent = speech_conditioning_latent + emo_vec.unsqueeze(1)
        conds_latent = torch.cat((conds_latent, conds_latent.new_zeros(conds_latent.size(0), 2, conds_latent.size(2))), 1)
        input_ids, inputs_embeds, attention_mask = self.prepare_gpt_inputs(conds_latent, text_inputs, langs)
        self.inference_model.store_mel_emb(inputs_embeds)
        trunc_index = input_ids.shape[1]
        logits_processor = LogitsProcessorList()
        if typical_sampling:
            # employ custom typical sampling
            if not (typical_mass > 0.0 and typical_mass < 1.0):
                raise ValueError(f"`typical_mass` has to be a float > 0 and < 1, but is {typical_mass}")
            min_tokens_to_keep = 2 if hf_generate_kwargs.get("num_beams", 1) > 1 else 1
            logits_processor.append(TypicalLogitsWarper(mass=typical_mass, min_tokens_to_keep=min_tokens_to_keep))
        max_length = trunc_index + (self.max_mel_tokens - 1 if max_generate_length is None else max_generate_length)

        if self.accel_engine is not None:
            with torch.cuda.device(input_ids.device):
                output = self.accel_engine.generate(
                    input_ids,  # fake input_ids (all 1s + start_mel_token)
                    max_new_tokens=max_length - trunc_index,
                    # do_sample=False is greedy, as in transformers
                    temperature=hf_generate_kwargs.get('temperature', 1) if hf_generate_kwargs.get('do_sample', True) else 0,
                    top_k=hf_generate_kwargs.get('top_k', 0),
                    top_p=hf_generate_kwargs.get('top_p', 1.0),
                    repetition_penalty=hf_generate_kwargs.get('repetition_penalty', 1.0),
                    stop_tokens=[self.stop_mel_token],
                    tts_embeddings=inputs_embeds,  # [pad][cond][text], without the start_mel_token
                    tts_mel_embedding=self.inference_model.embeddings,
                    tts_text_pos_embedding=self.inference_model.text_pos_embedding,
                )
        else:
            output = self.inference_model.generate(input_ids,
                                                bos_token_id=self.start_mel_token, pad_token_id=self.stop_mel_token,
                                                eos_token_id=self.stop_mel_token, attention_mask=attention_mask,
                                                max_length=max_length, logits_processor=logits_processor,
                                                num_return_sequences=1,
                                                **hf_generate_kwargs)
        return output[:, trunc_index:]

    def get_emo_conditioning(self, speech_conditioning_input, cond_mel_lengths=None):
        speech_conditioning_input, mask = self.emo_conditioning_encoder(speech_conditioning_input.transpose(1, 2),
                                                                        cond_mel_lengths)  # (b, s, d), (b, 1, s)
        conds_mask = self.emo_cond_mask_pad(mask.squeeze(1))
        conds = self.emo_perceiver_encoder(speech_conditioning_input, conds_mask)  # (b, 1, d)
        return conds.squeeze(1)

    def get_emovec(self, emo_speech_conditioning_latent, emo_cond_lengths):
        emo_vec_syn_ori = self.get_emo_conditioning(emo_speech_conditioning_latent.transpose(1,2), emo_cond_lengths)
        emo_vec_syn = self.emovec_layer(emo_vec_syn_ori)
        emo_vec = self.emo_layer(emo_vec_syn)
        return emo_vec

    def merge_emovec(self, speech_conditioning_latent, emo_speech_conditioning_latent, cond_lengths, emo_cond_lengths, alpha = 1.0):
        emo_vec = self.get_emovec(emo_speech_conditioning_latent, emo_cond_lengths)
        base_vec = self.get_emovec(speech_conditioning_latent, cond_lengths)

        out = base_vec + alpha * (emo_vec - base_vec)
        return out
