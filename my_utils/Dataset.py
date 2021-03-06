import torch
import os
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
#from PIL import Image


def default_loader(path):
    return read_image(path, mode=ImageReadMode.RGB)
    #return Image.open(path).convert('RGB')

class MyDataset(Dataset):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    def __init__(self, root, device, image_folder, label_folder, transform=None, target_transform=None, loader=default_loader):
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
        self.device = device

    def __getitem__(self, item):
        #torch.set_printoptions(profile="full")
        imageName = self.images[item]
        image = self.loader(imageName)
        image = image.to(self.device)
        
        if self.transform is not None:
            image = self.transform(image)   
        
        labels = self.labels[item]
        labels = torch.Tensor(labels)
        '''
        #****two head set****#
        labels_angle = self.labels_angle[item]
        labels_distance = self.labels_distance[item]
        '''
        '''
        from PIL import Image
        from torchvision import transforms
        image = transforms.ToPILImage()(image)
        image.show()
        '''
        return image, labels

    def __len__(self):
        return len(self.images)
        