from os.path import join


class Config(object):
    def __init__(self):

        # Dataset
        self.dataset = 'refcoco'
        self.ref_base = 'PATH'
        self.coco_dir = 'PATH'
        self.paco_base = 'PATH'
        self.paco_imgs = 'PATH'
        self.use_normalized_paco = True
        self.ref_dir = join(self.ref_base, self.dataset)
        self.limit = -1

        # Training Settings
        self.checkpoint_dir = './checkpoints'
        self.verbose = True
        self.epochs = 30
        self.save_every = 'max'
        self.prefix_length = 21  # valid length: each x for which (x % 2) - 1
        self.prefix_length_clip = 21
        self.batch_size = 16
        self.only_prefix = False
        self.output_prefix = self.dataset + '_prefix' if self.only_prefix else self.dataset + '_full'
        self.stop_after_epochs = 3

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