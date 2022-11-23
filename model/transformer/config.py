class Config():
    def __init__(self):
        self.num_heads = 2
        self.num_blocks = 2

        # embedding size
        self.d_model = 64

        self.max_seq_length = 32

        self.batch_size = 1

        self.ff_units = 512
        
        self.dropout_val = 0.1
        
        self.vocab_size = 6