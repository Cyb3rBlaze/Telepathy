class Config():
    def __init__(self):
        self.num_heads = 12
        self.num_blocks = 12
        self.d_model = 768
        self.batch_size = 1
        self.max_seq_length = 32
        self.ff_units = 512
        self.dropout_val = 0.1
        self.vocab_size = 6
        self.n_positions = 1024