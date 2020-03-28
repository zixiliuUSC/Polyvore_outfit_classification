import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from itertools import combinations
import os
import numpy as np
import os.path as osp
import json
from tqdm import tqdm
from PIL import Image

from utils import Config


class polyvore_dataset:
    def __init__(self):
        self.root_dir = Config['root_path']
        self.image_dir = osp.join(self.root_dir, 'images')
        self.transforms = self.get_data_transforms()
        # self.X_train, self.X_test, self.y_train, self.y_test, self.classes = self.create_dataset()
        # map id to category
        meta_file = open(osp.join(self.root_dir, Config['meta_file']), 'r')
        meta_json = json.load(meta_file)
        id_to_category = {}
        for k, v in meta_json.items():
            id_to_category[k] = v['category_id']
        # create X, y pairs
        files = os.listdir(self.image_dir)
        X = []; y = []
        for x in files:
            # x[:-4] is to avoid parsing ".jpg"
            if x[:-4] in id_to_category:
                X.append(x)
                y.append(int(id_to_category[x[:-4]]))

        label_encode = LabelEncoder().fit(y)
        y = label_encode.transform(y)
        
        id_to_category = {}
        for i in range(len(X)):
            id_to_category[X[i][:-4]] = y[i]
        comb = list(combinations(list(np.arange(max(y)+1)),2))
        # map outfit id to item id (train)
        meta_outfit_train = open(osp.join(self.root_dir,'train.json'),'r')
        meta_outfit_train_json = json.load(meta_outfit_train)
        style_to_id_train = dict()
        dataset_train = []
        #y_train = []
        for j in meta_outfit_train_json:
            style_to_id_train[j['set_id']] = []
            for i in j['items']:
                #r = int(id_to_category[i['item_id']])
                #print(r)
                #label_encode.transform(r)
                #style_to_id_train[j['set_id']].append((i['item_id'],label_encode.transform([int(id_to_category[i['item_id']])]).item()))
                style_to_id_train[j['set_id']].append(i['item_id'])
        
        with open(osp.join(self.root_dir,'compatibility_train.txt'),'r') as f:
            for line in f:
                line = line.split()
                outfit = []
                for i in line[1:]:
                    cloth = i.split('_')
                    cloth[1] = int(cloth[1])-1
                    outfit.append(style_to_id_train[cloth[0]][cloth[1]])
                for j in combinations(outfit,2):
                    dataset_train.append([j,int(line[0])])
                    #y_train.append(int(line[0]))
        comb_dic = dict()
        X_train_category = dict()
        for i in comb:
            comb_dic[i]=0
            X_train_category[i]=[]
    
        for i in range(153):
            comb_dic[(i,i)]=0
            X_train_category[(i,i)]=[]

        for t,i in enumerate(dataset_train):
            k = id_to_category[i[0][0]]
            v = id_to_category[i[0][1]]
            if k>=v:
                comb_dic[(v,k)] += 1
                X_train_category[(v,k)].append([i[0],i[1]])
            else:
                comb_dic[(k,v)] += 1
                X_train_category[(k,v)].append([i[0],i[1]])
        self.comb_dic = comb_dic
        self.X_train_category = X_train_category

        dataset_test = []
        # map outfit id to item id (val)
        meta_outfit_val = open(osp.join(self.root_dir,'valid.json'),'r')
        meta_outfit_val_json = json.load(meta_outfit_val)
        style_to_id_val = dict()
        for j in meta_outfit_val_json:
            style_to_id_val[j['set_id']] = []
            for i in j['items']:
                style_to_id_val[j['set_id']].append(i['item_id'])
        with open(osp.join(self.root_dir,'compatibility_valid.txt'),'r') as f:
            for line in f:
                line = line.split()
                outfit = []
                for i in line[1:]:
                    cloth = i.split('_')
                    cloth[1] = int(cloth[1])-1
                    outfit.append(style_to_id_val[cloth[0]][cloth[1]])
                for j in combinations(outfit,2):
                    dataset_test.append([j,int(line[0])])

        comb_dic = dict()
        X_train_category = dict()
        for i in comb:
            comb_dic[i]=0
            X_train_category[i]=[]
    
        for i in range(153):
            comb_dic[(i,i)]=0
            X_train_category[(i,i)]=[]

        for t,i in enumerate(dataset_test):
            k = id_to_category[i[0][0]]
            v = id_to_category[i[0][1]]
            if k>=v:
                comb_dic[(v,k)] += 1
                X_train_category[(v,k)].append([i[0],i[1]])
            else:
                comb_dic[(k,v)] += 1
                X_train_category[(k,v)].append([i[0],i[1]])
        self.comb_dic_test = comb_dic
        self.X_test_category = X_train_category


    def get_data_transforms(self):
        data_transforms = {
            'train': transforms.Compose([
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ]),
            'test': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ]),
        }
        return data_transforms



    def create_dataset(self):
        '''
        #return label_encode
        print('len of X: {}, # of categories: {}'.format(len(X), max(y) + 1))
        '''
        dataset_train_reduce = []
        for key, value in self.comb_dic.items():
            clothes = self.X_train_category[key]
            idx = random.sample(list(np.arange(value)),round(0.6*value))
            if idx!=[]:
                for i in idx:
                    dataset_train_reduce.append(self.X_train_category[key][i]) 
        dataset_train_reduce = np.array(dataset_train_reduce)
        
        
        dataset_test_reduce = []
        for key, value in self.comb_dic_test.items():
            clothes = self.X_test_category[key]
            idx = random.sample(list(np.arange(value)),value)
            if idx!=[]:
                for i in idx:
                    dataset_test_reduce.append(self.X_test_category[key][i]) 
        dataset_test_reduce = np.array(dataset_test_reduce)
        #print(style_to_label_train)
        #X_train = []
        #y_train = []
        #for key in style_to_label_train:
            #for i in combinations(style_to_id_train[key],2):
                #X_train.append(i)
                #y_train.append(style_to_label_train[key])
            #instance = list(combinations(style_to_id_train[key],2))
            #y_instance = [style_to_label_train[key] for i in range(len(instance))]
            #X_train.append(instance)
            #y_train.append(y_instance)

        # split dataset
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        #return X_train, X_test, y_train, y_test, max(y) + 1
        return dataset_train_reduce[:,0], dataset_test_reduce[:,0], dataset_train_reduce[:,1].astype('int'), dataset_test_reduce[:,1].astype('int'),2



# For category classification
class polyvore_train(Dataset):
    def __init__(self, X_train, y_train, transform):
        self.X_train = X_train
        self.y_train = y_train
        self.transform = transform
        self.image_dir = osp.join(Config['root_path'], 'images')

    def __len__(self):
        return len(self.X_train)

    def __getitem__(self, item):
        filename1 = self.X_train[item][0]+'.jpg'
        filename2 = self.X_train[item][1]+'.jpg'
        file_path1 = osp.join(self.image_dir, filename1)
        file_path2 = osp.join(self.image_dir, filename2)
        return self.transform(Image.open(file_path1)),self.transform(Image.open(file_path2)),self.y_train[item]




class polyvore_test(Dataset):
    def __init__(self, X_test, y_test, transform):
        self.X_test = X_test
        self.y_test = y_test
        self.transform = transform
        self.image_dir = osp.join(Config['root_path'], 'images')


    def __len__(self):
        return len(self.X_test)


    def __getitem__(self, item):
        filename1 = self.X_test[item][0]+'.jpg'
        filename2 = self.X_test[item][1]+'.jpg'
        file_path1 = osp.join(self.image_dir, filename1)
        file_path2 = osp.join(self.image_dir, filename2)
        #print(item)
        #print(file_path1)
        #print('----------')
        #print(file_path2)
        return self.transform(Image.open(file_path1)), self.transform(Image.open(file_path2)),self.y_test[item]




def get_dataloader(debug, batch_size, num_workers):
    dataset = polyvore_dataset()
    transforms = dataset.get_data_transforms()
    X_train, X_test, y_train, y_test, classes = dataset.create_dataset()

    if debug==True:
        train_set = polyvore_train(X_train[:100], y_train[:100], transform=transforms['train'])
        test_set = polyvore_test(X_test[:100], y_test[:100], transform=transforms['test'])
        dataset_size = {'train': len(y_train), 'test': len(y_test)}
    else:
        train_set = polyvore_train(X_train, y_train, transforms['train'])
        test_set = polyvore_test(X_test, y_test, transforms['test'])
        dataset_size = {'train': len(y_train), 'test': len(y_test)}

    datasets = {'train': train_set, 'test': test_set}
    dataloaders = {x: DataLoader(datasets[x],
                                 shuffle=True if x=='train' else False,
                                 batch_size=batch_size,
                                 num_workers=num_workers)
                                 for x in ['train', 'test']}
    return dataloaders, classes, dataset_size




########################################################################
# For Pairwise Compatibility Classification

