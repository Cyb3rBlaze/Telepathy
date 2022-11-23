class Config():
    def __init__(self):
        self.num_blocks = 3

        self.batch_size = 1

        self.in_channels = 1
        
        self.channel_multiplier = 16

        self.pool_stride = 2
        self.pool_kernel = 2

        self.conv_kernel = 3