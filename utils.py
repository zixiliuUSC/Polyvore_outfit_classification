import numpy as np
import os
import os.path as osp
import argparse

Config ={}
Config['root_path'] = '/mnt/d/EE-599/polyvore_outfits'
Config['meta_file'] = 'polyvore_item_metadata.json'
Config['checkpoint_path'] = 'checkpoint_OHEM'


Config['use_cuda'] = True
Config['debug'] = True
Config['num_epochs'] = 25
Config['batch_size'] = 2

Config['learning_rate'] = 0.001
Config['num_workers'] = 8
Config['lr_decay'] = 1

