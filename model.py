#import sys
#print(sys.path)
#sys.path.append('/nas/home/zixiliu/anaconda3/envs/pytorch/lib/python3.7/site-packages')
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2

class fashion_model(nn.Module):
    def __init__(self, class_num):
        super(fashion_model, self).__init__()
        self.model = mobilenet_v2(pretrained=True).features
        child_counter = 0
        for child in self.model.children():
            if child_counter < 16:
                print("child ",child_counter," was frozen")
                child_counter += 1
                for param in child.parameters():
                    param.requires_grad = False
            else:
                print("child ",child_counter," was not frozen")
                child_counter += 1
        num_ftrs = mobilenet_v2(pretrained=True).classifier[1].in_features
        self.classifier = nn.Sequential(nn.Dropout(0.2),nn.Linear(num_ftrs*2,100,True),nn.ReLU(),nn.Linear(100,2,True))
        self.init_linear()

    def init_linear(self):
        nn.init.xavier_uniform_(self.classifier[1].weight)
        nn.init.xavier_uniform_(self.classifier[3].weight)

    def forward(self, input1, input2):
        emb1 = self.model(input1)
        emb2 = self.model(input2)
        emb1 = nn.functional.adaptive_avg_pool2d(emb1, 1).reshape(emb1.shape[0], -1)
        emb2 = nn.functional.adaptive_avg_pool2d(emb2, 1).reshape(emb2.shape[0], -1)
        emb = torch.cat((emb1,emb2),1)
        output = self.classifier(emb)
        return output


#nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
'''
child_counter = 0
for child in model.children():
    if child_counter < 7:
        print("child ",child_counter," was frozen")
        child_counter += 1
        for param in child.parameters():
            param.requires_grad = False
    elif child_counter == 7:
        children_of_child_counter = 0
        for children_of_child in child.children():
            for param in children_of_child.parameters():
                #param.requires_grad = False
                print('child ', children_of_child_counter, 'of child',child_counter,' was not frozen')
            children_of_child_counter += 1
            #if children_of_child_counter > 1:
                #for param in children_of_child.parameters():
                    #param.requires_grad = False
                    #print('child ', children_of_child_counter, 'of child',child_counter,' was frozen')
                #children_of_child_counter += 1
            #else:
                #print('child ', children_of_child_counter, 'of child',child_counter,' was not frozen')
                #children_of_child_counter += 1
    else:
        print("child ",child_counter," was not frozen")
        child_counter += 1
'''
