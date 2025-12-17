import numpy as np


class Linear(object):

    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim

    def num_parameters(self):
        return self.input_dim * self.output_dim

    def flops(self, seq_len):
        return 6 * seq_len * self.input_dim * self.output_dim


class Attention(object):

    def __init__(self, seq_len, model_dim, num_heads):
        self.seq_len = seq_len
        self.model_dim = model_dim
        self.num_heads = num_heads

        self.qkv = Linear(model_dim, 3 * model_dim)
        self.out = Linear(model_dim, model_dim)

    def num_parameters(self):
        return self.qkv.num_parameters() + self.out.num_parameters()

    def flops(self):
        total_flops = self.qkv.flops(self.seq_len)  # qkv linear
        total_flops += 6 * (self.seq_len ** 2) * self.model_dim  # QK^T
        total_flops += 9 * (self.seq_len ** 2) * self.num_heads  # softmax
        total_flops += 6 * (self.seq_len ** 2) * self.model_dim  # softmax(A)V
        total_flops += self.out.flops(self.seq_len)
        return total_flops


class FFN(object):

    def __init__(self, model_dim, hidden_dim, swiglu):
        self.model_dim = model_dim
        self.hidden_dim = hidden_dim
        self.h1 = Linear(model_dim, hidden_dim)
        self.h2 = Linear(hidden_dim, model_dim)
        self.h3 = Linear(model_dim, hidden_dim) if swiglu else None

    def num_parameters(self):
        params = self.h1.num_parameters() + self.h2.num_parameters()
        if self.h3 is not None:
            params += self.h3.num_parameters()
        return params

    def flops(self, seq_len):
        total_flops = self.h1.flops(seq_len)  # h1
        total_flops += self.hidden_dim  # non-linear function
        total_flops += self.h2.flops(seq_len)  # h2
        if self.h3 is not None:
            total_flops += self.h3.flops(seq_len)  # h3

        return total_flops


class MoE(object):

    def __init__(self, model_dim, hidden_dim, swiglu, num_experts, topk):
        self.ffn = FFN(model_dim, hidden_dim, swiglu)
        self.num_experts = num_experts
        self.topk = topk

    def num_parameters(self):
        return self.num_experts * self.ffn.num_parameters()

    def flops(self, seq_len):
        return self.topk * self.ffn.flops(seq_len)


class TransformerLayer(object):

    def __init__(self, seq_len, model_dim, hidden_dim, num_heads, swiglu, num_experts, topk):
        self.seq_len = seq_len
        self.attention = Attention(seq_len, model_dim, num_heads)
        self.moe = MoE(model_dim, hidden_dim, swiglu, num_experts, topk)

    def num_parameters(self):
        return self.attention.num_parameters() + self.moe.num_parameters()

    def flops(self):
        return self.attention.flops() + self.moe.flops(self.seq_len)


class Transformer(object):

    def __init__(self, num_layers, seq_len, vocab_size, model_dim, hidden_dim, num_heads, swiglu, num_experts, topk):
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.embed = Linear(vocab_size, model_dim)
        self.layer = TransformerLayer(seq_len, model_dim, hidden_dim, num_heads, swiglu, num_experts, topk)
        self.logit = Linear(model_dim, vocab_size)

    def num_parameters(self):
        params = self.embed.num_parameters()
        params += self.num_layers * self.layer.num_parameters()
        params += self.logit.num_parameters()
        return params
    
    def num_parameters_no_embed(self):
        return self.num_layers * self.layer.num_parameters()

    def flops(self):
        # return self.embed.flops(self.seq_len) + self.num_layers * self.layer.flops() + self.logit.flops(self.seq_len)
        return self.num_layers * self.layer.flops()

def main():
    num_layers = 24
    seq_len = 8192
    vocab_size = 250240
    model_dim = 2048
    hidden_dim = 5376
    num_heads = 16
    num_experts = 1
    topk = 1
    swiglu = True
    transformer = Transformer(num_layers, seq_len, vocab_size, model_dim, hidden_dim, num_heads, swiglu, num_experts, topk)
    print(f"num params: {transformer.num_parameters() / 1e6:.2f}M")
    flops_per_seq = transformer.flops()
    print(f"flops: {flops_per_seq:.2e} per {seq_len} tokens")
    total_tokens = 1135*1e9  # 343B
    print(f"total flops for {total_tokens / 1e9:.2f}B tokens: {flops_per_seq * (total_tokens / seq_len):.2e}")
    print(f"ratio: {transformer.flops() / (transformer.num_parameters_no_embed() * 6 * seq_len):.2f}")


if __name__ == '__main__':
    main()
