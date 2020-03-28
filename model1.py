import sys
print(sys.path)
sys.path.append('/nas/home/zixiliu/anaconda3/envs/pytorch/lib/python3.7/site-packages')
from torchvision.models import resnet50

model = resnet50(pretrained=True)

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