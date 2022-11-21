class Config():
    def __init__(self):
        self.num_heads = 2
        self.d_model = 16
        self.batch_size = 4
        self.max_seq_length = 32
        self.ff_units = 512
        self.dropout_val = 0.1
        self.vocab_size = 2000