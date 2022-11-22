import math

import torch
from torch import nn


# pulled from Dr. Karpathy's minGPT implementation
class GELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


# ScaledDotProductAttention for producing attention weighted output vectors
class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_keys, max_seq_length, masked=False):
        super().__init__()

        self.d_keys = d_keys
        
        self.masked = masked
        if self.masked:
            self.max_seq_length = max_seq_length
            self.mask = torch.tril(torch.ones(self.max_seq_length, self.max_seq_length)).view(1, 1, self.max_seq_length, self.max_seq_length)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        queries, keys, values = x

        seq_length = queries.shape[2]

        qk_matrix = torch.matmul(queries, torch.transpose(keys, -1, -2))
        attention_pattern = qk_matrix/math.sqrt(self.d_keys)

        if self.masked:
            attention_pattern = attention_pattern.masked_fill(self.mask[:,:,:seq_length, :seq_length] == 0, float('-inf'))

        attention_pattern = self.softmax(attention_pattern)

        return torch.matmul(attention_pattern, values)


# MultiHeadAttention class for processing different parts of input embedded matrix
class MultiHeadAttention(nn.Module):
    def __init__(self, config, masked=False):
        super().__init__()

        self.num_heads = config.num_heads
        self.d_model = config.d_model
        self.d_keys = self.d_model//self.num_heads
        self.batch_size = config.batch_size

        self.query_params = nn.Linear(self.d_model, self.d_model)
        self.key_params = nn.Linear(self.d_model, self.d_model)
        self.value_params = nn.Linear(self.d_model, self.d_model)

        # dims has to be divisible by num_heads
        self.scaled_dot_product_attention = ScaledDotProductAttention(self.d_model/self.num_heads, config.max_seq_length, masked)
        self.output_linear_params = nn.Linear(self.d_model, self.d_model)

        # value referenced from Justin's implementation
        self.dropout = nn.Dropout(config.dropout_val)

    def forward(self, x):
        seq_length = x.shape[1]

        projected_queries = self.query_params(x).reshape((self.batch_size, seq_length, self.num_heads, self.d_keys))
        projected_keys = self.key_params(x).reshape((self.batch_size, seq_length, self.num_heads, self.d_keys))
        projected_values = self.value_params(x).reshape((self.batch_size, seq_length, self.num_heads, self.d_keys))

        # changing dims to be (batch_size, num_heads, sequence_length, d_keys)
        transposed_queries = torch.transpose(projected_queries, -2, -3)
        transposed_keys = torch.transpose(projected_keys, -2, -3)
        transposed_values = torch.transpose(projected_values, -2, -3)

        attention_weighted_output = self.scaled_dot_product_attention((transposed_queries, transposed_keys, transposed_values)).reshape((self.batch_size, seq_length, self.d_model))

        head_output = self.output_linear_params(attention_weighted_output)

        return self.dropout(head_output)


# FeedForward class for MLP component after attention computations
class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.linear1 = nn.Linear(config.d_model, config.ff_units)
        self.linear2 = nn.Linear(config.ff_units, config.d_model)

        self.gelu = GELU()

        # value referenced from minGPT implementation
        self.dropout = nn.Dropout(config.dropout_val)

    def forward(self, x):
        hidden_output1 = self.linear1(x)
        hidden_activation1 = self.gelu(hidden_output1)
        hidden_output2 = self.linear2(hidden_activation1)

        return self.dropout(hidden_output2)


# EncoderBlock class wrapping encoder attention + MLP components
class EncoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.multi_head_attention = MultiHeadAttention(config)

        self.layer_norm = nn.LayerNorm(config.d_model)

        self.feed_forward = FeedForward(config)

    def forward(self, x):
        # queries, keys, and values stem from same input sequence -> self-attention
        attention_output = self.multi_head_attention(x)
        residual_stream = torch.add(x, self.layer_norm(attention_output))

        feed_forward_output = self.feed_forward(residual_stream)

        return torch.add(residual_stream, self.layer_norm(feed_forward_output))


# DecoderBlock class wrapping decoder attention + MLP components
class DecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.masked_multi_head_attention = MultiHeadAttention(config, masked=True)
        self.multi_head_attention = MultiHeadAttention(config)

        self.layer_norm = nn.LayerNorm(config.d_model)

        self.feed_forward = FeedForward(config)

    def forward(self, x):
        # queries, keys, and values stem from same input sequence -> self-attention
        masked_attention_output = self.masked_multi_head_attention(x)
        residual_stream = torch.add(x, self.layer_norm(masked_attention_output))

        attention_output = self.multi_head_attention(residual_stream)
        residual_stream = torch.add(residual_stream, self.layer_norm(attention_output))

        feed_forward_output = self.feed_forward(residual_stream)

        return torch.add(residual_stream, self.layer_norm(feed_forward_output))


# Transformer class wrapping all internal components
class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        

        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        # initialize trainable params to matrix of zeros
        self.positional_encodings = nn.Parameter(torch.zeros(1, config.n_positions, config.d_model))

        # value referenced from Justin's implementation
        self.encoding_dropout = nn.Dropout(config.dropout_val)

        self.decoder_blocks = []
        for i in range(config.num_blocks):
            self.decoder_blocks += [DecoderBlock(config)]

        self.unembedding_matrix = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.softmax = nn.Softmax(dim=-1)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x):
        # dims[0] = batch size, dims[1] = embedding size = d_model, dims[2] = number of tokens
        embedded_input = self.embedding(x)
        positional_encodings = self.positional_encodings[:, :x.shape[1], :]

        encoded_input = self.encoding_dropout(embedded_input + positional_encodings)

        decoder_output = self.decoder_blocks[0](encoded_input)
        for i in range(1, len(self.decoder_blocks)):
            decoder_output = self.decoder_blocks[i](decoder_output)

        unembedded_output = self.unembedding_matrix(decoder_output)

        return self.softmax(unembedded_output)