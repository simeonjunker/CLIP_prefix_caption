from os.path import join


class Config(object):
    def __init__(self):

        # Dataset
        self.dataset = 'refcoco'
        self.coco_dir = 'PATH'
        self.ref_base = 'PATH'
        self.ref_dir = join(self.ref_base, self.dataset)
        self.limit = -1

        # Training Settings
        self.only_prefix = False
        self.checkpoint_dir = './checkpoints'
        self.output_prefix = f'{self.dataset}_{"prefix" if self.only_prefix else "full"}'
        self.verbose = True
        self.epochs = 10
        self.save_every = 1
        self.prefix_length = 11  # valid length: each x for which (x % 2) - 1 == 0
        self.prefix_length_clip = 11
        self.batch_size = 2

        # Model Settings
        self.use_global_features = True
        self.use_location_features = True
        self.max_length = 128
        self.mapping_type = 'mlp'
        self.num_layers = 8
        self.is_rn = False
        self.normalize_prefix = False

if __name__ == '__main__':
    
    config = Config()
    
    print('### CONFIGURATION ###')
    for key, value in vars(config).items():
        print(f'# {key}: {value}')
    print('#####################')