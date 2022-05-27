import torch
import os
from torch.utils.data import Dataset
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def default_loader(path):
    return Image.open(path).convert('RGB')

class MyDataset(Dataset):
    def __init__(self, root, image_folder, label_folder, transform=None, target_transform=None, loader=default_loader):
        super(Dataset, self).__init__()
        images = []
        labels = []
        '''
        #****two head set****#
        labels_angle = []
        labels_distance = []
        '''
        
        #####image#####
        All_image_name = os.listdir(os.path.join(root,image_folder))
        for Image_name in All_image_name:
            images.append(os.path.join(root,image_folder,Image_name))
            #####label#####
            Label_name = Image_name.split('.')
            Label_name = Label_name[0]+'.txt'
            f = open(os.path.join(root,label_folder,Label_name), 'r')
            firstline = f.readline().rstrip()
            firstline = firstline.split(' ')
            labels.append([float(firstline[0]),float(firstline[1])])
            '''
            #****two head set****#
            labels_angle.append(float(firstline[0]))
            labels_distance.append(float(firstline[1]))
            '''

        self.images = images
        self.labels = labels
        '''
        #****two head set****#
        self.labels_angle = labels_angle
        self.labels_distance = labels_distance
        '''
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, item):
        imageName = self.images[item]
        image = self.loader(imageName)
        if self.transform is not None:
            image = self.transform(image)
        ##print(imageName)
        labels = self.labels[item]
        labels = torch.Tensor(labels)
        '''
        #****two head set****#
        labels_angle = self.labels_angle[item]
        labels_distance = self.labels_distance[item]
        '''
        return image, labels

    def __len__(self):
        return len(self.images)
        