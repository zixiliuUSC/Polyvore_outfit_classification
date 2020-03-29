import numpy as np
import os
import os.path as osp
import argparse

Config ={}
Config['root_path'] = '/home/ubuntu/HW4/polyvore_outfits'
Config['meta_file'] = 'polyvore_item_metadata.json'
Config['checkpoint_path'] = 'checkpoint'


Config['use_cuda'] = True
Config['debug'] = True
Config['num_epochs'] = 11
Config['batch_size'] = 125

Config['learning_rate'] = 0.001
Config['num_workers'] = 8
Config['lr_decay'] = 0.98

