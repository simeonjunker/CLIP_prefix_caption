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
        self.checkpoint_dir = './checkpoints'
        self.output_prefix = 'refcoco_prefix'
        self.verbose = True
        self.epochs = 10
        self.save_every = 1
        self.prefix_length = 10
        self.prefix_length_clip = 10
        self.batch_size = 2
        self.only_prefix = False

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