
import torch
from torch import nn

from torch.nn.attention.flex_attention import flex_attention, create_block_mask, and_masks, or_masks

from attn_gym import visualize_attention_scores

import seaborn as sns
import matplotlib.pyplot as plt


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class SWA_SinkMeta(nn.Module):

    def __init__(self, 
        num_attention_heads, 
        num_key_value_heads, 
        attention_head_size, 
        attention_window_size=None, 
        num_meta_tokens=None, 
        seq_length=None, 
        use_positional_embedding=False, 
        rope_base=None):
        super().__init__()

        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.attention_head_size = attention_head_size
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads 

        self.attention_window_size = attention_window_size

        self.num_meta_tokens = num_meta_tokens
        self.seq_length = seq_length

        self.use_positional_embedding = use_positional_embedding
        self.rope_base = rope_base

        # when using sliding window attention with meta token, we modify the attention mask to 
        # for example, when window_size = 3, num_meta_tokens = 2, the attention mask becomes 
        # 1
        # 1 1
        # 1 1 1
        # 1 1 1 1
        # 1 1 1 1 1
        # 1 1 0 1 1 1
        # 1 1 0 0 1 1 1
        # 1 1 0 0 0 1 1 1

        # in order to support the modified attention mask, we have to use flexattention (PyTorch>=2.5.0) instead of flash_attention

        # precompile the attention mask for efficiency purposes 
        def sliding_window(b, h, q_idx, kv_idx):
            return q_idx - kv_idx <= self.attention_window_size

        def causal_mask(b, h, q_idx, kv_idx):
            return q_idx >= kv_idx
        attn_mask = and_masks(causal_mask, sliding_window)
    
        def prefix_mask(b, h, q_idx, kv_idx):
            return kv_idx < self.num_meta_tokens
        register_mask = and_masks(causal_mask, prefix_mask)

        self.attn_mask = or_masks(attn_mask, register_mask) # real mask we use 

        qk_length = self.seq_length + self.num_meta_tokens

        self.block_mask = create_block_mask(
            self.attn_mask, 
            B=None, H=None, Q_LEN=qk_length, KV_LEN=qk_length)

        self.flex_attention = torch.compile(flex_attention)


        if self.use_positional_embedding:
            self.rotary_emb = RotaryEmbedding(
                dim=self.attention_head_size,
                base=self.rope_base)


    

    def forward(self, query_states, key_states, value_states):
        bsz, q_len, _ = query_states.size()

        query_states = query_states.view(bsz, q_len, self.num_attention_heads, self.attention_head_size).transpose(1, 2).contiguous()

        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.attention_head_size).transpose(1, 2).contiguous()

        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.attention_head_size).transpose(1, 2).contiguous()

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        if self.use_positional_embedding:
            cos, sin = self.rotary_emb(query_states)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)


        if key_states.shape[-2] <= self.block_mask.shape[-2] - 128 or key_states.shape[-2] > self.block_mask.shape[-2]:
            # 128 is the minimum block size for flex_attention
            block_mask = create_block_mask(self.attn_mask, B=None, H=None, Q_LEN=key_states.shape[-2], KV_LEN=key_states.shape[-2])

        else:
            # reuse the mask if possible 
            block_mask = self.block_mask

        attn_outputs = self.flex_attention(query_states, key_states, value_states, block_mask=block_mask)

        attn_outputs = attn_outputs.transpose(1, 2).contiguous()
        attn_outputs = attn_outputs.reshape(bsz, q_len, int(self.num_attention_heads * self.attention_head_size)).contiguous()

        return attn_outputs

        
if __name__ == "__main__":
    device = "cuda"
    configs = {
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "attention_head_size": 128,
        "attention_window_size": 128,
        "num_meta_tokens": 32,
        "seq_length": 1024,
        "use_positional_embedding": False,
        "rope_base": 150000,
    }

    layer = SWA_SinkMeta(**configs)


    query_states = torch.ones(1, configs["seq_length"], configs["num_attention_heads"], configs["attention_head_size"])
    key_states = torch.ones(1, configs["seq_length"], configs["num_attention_heads"], configs["attention_head_size"])


    meta_tokens_key = torch.ones(1, configs["num_meta_tokens"], configs["num_attention_heads"], configs["attention_head_size"])
    meta_tokens_query = torch.ones(1, configs["num_meta_tokens"], configs["num_attention_heads"], configs["attention_head_size"])


    key_states = torch.cat([meta_tokens_key, key_states], dim=1).to(device).transpose(1, 2)
    query_states = torch.cat([meta_tokens_query, query_states], dim=1).to(device).transpose(1, 2)

    # attn_outputs = layer(query_states, key_states, value_states)
    visualize_attention_scores(
        query_states, key_states, mask_mod=layer.attn_mask, device=device, name="sliding_window_mask"
    )
    