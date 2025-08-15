from __future__ import annotations
from dataclasses import asdict, dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from qonnx.util.cleanup import cleanup
import onnx

def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False)->torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value


class CustomMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(CustomMultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def split_heads(self, x):
        batch_size, seq_length, _ = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def forward(self, query, key, value):
        batch_size = query.size(0)

        # Linear projections
        Q = self.split_heads(self.W_q(query))
        K = self.split_heads(self.W_k(key))
        V = self.split_heads(self.W_v(value))

        # compute attention
        context = F.scaled_dot_product_attention(
            Q, K, V, dropout_p=0.0, scale=None
        )

        # Concatenate heads and apply final linear transformation
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(context)
        return output, None


class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_ff, num_heads):
        super().__init__()
        self.mha = CustomMultiHeadAttention(d_model=d_model, num_heads=num_heads)

        self.ffwd = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )
        self.prenorm_1 = nn.LayerNorm(normalized_shape=d_model)
        self.prenorm_2 = nn.LayerNorm(normalized_shape=d_model)

        self.skip_connection_norm_1 = nn.LayerNorm(normalized_shape=d_model)
        self.skip_connection_norm_2 = nn.LayerNorm(normalized_shape=d_model)

    def attention(self, member_embeddings, candidate_items) -> torch.Tensor:
        attn_output, _ = self.mha(member_embeddings, candidate_items, candidate_items)
        return attn_output

    def forward(self, member_embeddings, candidate_items) -> torch.Tensor:
        attn = self.attention(self.prenorm_1(member_embeddings), candidate_items)
        x = self.skip_connection_norm_1(member_embeddings + attn)

        ffwd = self.ffwd(self.prenorm_2(x))
        output = self.skip_connection_norm_2(x + ffwd)
        return output


class GenerativeRecommender(nn.Module):
    def __init__(self, sequence_length, d_model, dff_multiplier, d_labels, num_heads, num_layers):
        super().__init__()
        self.sequence_length = sequence_length
        self.d_model = d_model

        self.register_buffer("sequence_indices", torch.arange(1, self.sequence_length + 1))

        transformer_layers = []
        for _ in range(num_layers):
            module = TransformerLayer(
                d_model=self.d_model,
                num_heads=num_heads,
                d_ff=int(self.d_model * dff_multiplier)
            )
            transformer_layers.append(module)

        self.transformer_layers = nn.ModuleList(transformer_layers)
        self.head = nn.Linear(self.d_model, d_labels)

    def forward(self, member_embeddings: torch.Tensor, candidate_items: torch.Tensor):
        # Process separately
        for layer in self.transformer_layers:
            member_embeddings = layer(member_embeddings, candidate_items)

        logits = self.head(member_embeddings)
        return logits

    def score(self, member_embeddings: torch.Tensor, candidate_items: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            logits = self.forward(member_embeddings=member_embeddings, candidate_items=candidate_items)

        logits = logits[:candidate_items.size(1), :]
        label_scores = torch.sigmoid(logits)
        return label_scores


@dataclass
class ModelSpec:
    sequence_length: int
    d_model: int
    num_layers: int
    num_heads: int
    dff_multiplier: int = 4
    d_labels: int = 2

if __name__ == "__main__":
    small_model_spec = ModelSpec(
        sequence_length=1024,
        d_model=256,
        num_layers=1,
        num_heads=4,
    )

    model_spec = small_model_spec
    model = GenerativeRecommender(**asdict(model_spec))

    num_candidates = 512
    member_embeddings = torch.randn(1, 2 * model_spec.sequence_length, model_spec.d_model)
    candidate_items = torch.randn(1, num_candidates, model_spec.d_model)

    params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {params:,}")

    # Export the model to ONNX format
    torch.onnx.export(
        model,
        (member_embeddings, candidate_items),
        "genrec.onnx",
        input_names=["member_embeddings", "candidate_items"],
        output_names=["logits"],
        opset_version=19
    )

    new_model = onnx.load("genrec.onnx")
    cleanup(in_file="genrec.onnx", out_file="genrec_cleanedup.onnx")
