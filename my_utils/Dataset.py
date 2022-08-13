import torch
import os
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
from PIL import Image


def default_loader(path):
    return read_image(path, mode=ImageReadMode.RGB)
    #return Image.open(path).convert('RGB')

class MyDataset(Dataset):
    #device = torch.device('cuda' if torch.cuda.is_targetvailable() else 'cpu')
    def __init__(self, root, device, image_folder, label_folder, transform=None,
                ori_transform=None, loader=default_loader, folder_label=0):
        super(Dataset, self).__init__()
        images = []
        images_target = []
        labels = []
        '''
        #****two head set****#
        labels_targetngle = []
        labels_distance = []
        '''
        
        #####image#####
        All_image_name = os.listdir(os.path.join(root,image_folder[0]))
        for Image_name in All_image_name:
            images.append(os.path.join(root,image_folder[0],Image_name))
            images_target.append(os.path.join(root,image_folder[1],Image_name))
            #####label#####
            Label_name = Image_name.split('.')
            Label_name = Label_name[0]+'.txt'
            f = open(os.path.join(root,label_folder,Label_name), 'r')
            firstline = f.readline().rstrip()
            firstline = firstline.split(' ')
            labels.append([float(firstline[0]),float(firstline[1])])
            '''
            #****two head set****#
            labels_targetngle.append(float(firstline[0]))
            labels_distance.append(float(firstline[1]))
            '''
        
        self.images = images
        self.images_target = images_target
        self.labels = labels
        '''
        #****two head set****#
        self.labels_targetngle = labels_targetngle
        self.labels_distance = labels_distance
        '''
        self.transform = transform
        self.ori_transform = ori_transform
        self.loader = loader
        self.device = device
        self.folder_label = folder_label

    def __getitem__(self, item):
        #torch.set_printoptions(profile="full")
        imageName = self.images[item]
        image = self.loader(imageName)
        image = image.to(self.device)
        
        imageName = self.images_target[item]
        image_target = self.loader(imageName)
        image_target = image_target.to(self.device)
        
        if self.transform is not None:
            image = self.ori_transform(image)  
            image_target = self.transform(image_target)
        labels_target = self.labels[item][self.folder_label]
        '''
        #****two head set****#
        labels_targetngle = self.labels_targetngle[item]
        labels_distance = self.labels_distance[item]
        '''
        '''
        from PIL import Image
        from torchvision import transforms
        image = transforms.ToPILImage()(image)
        image.show()
        '''
        #print(image)
        return image.float(), image_target.float(), labels_target

    def __len__(self):
        return len(self.images)
        