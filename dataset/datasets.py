
import os
import torch
from PIL import Image, ImageFile
from torchvision import transforms
import torchvision.datasets.folder
from torch.utils.data import TensorDataset, Subset, ConcatDataset, Dataset

# firstly define a abstract class for multi-domain-dataset
class MultiDomainDataset:
    # set the contributions that contains steps, domains and so on
    STEPS = 1500
    CHECKPOINT_FREQ = 100
    DOMAINS = None # subclasses should define this
    INPUT_SHAPE = None # subclasses should define this

    def __getitem__(self, idx):
        return self.dataset[idx]
    
    def __len__(self):
        return len(self.dataset)
    
# for PACS these kind of image folder dataset share the same structure,
#  so we can define a farther abstact class for them

# this kind of datasets must have the following structure:
# root(folder, often named as "datasets")
#     +---dataset_name(folder, ex. "PACS")
#                     +---domain(folder, ex. "art_painting")
#                               +---class(folder, ex. "dog")
#                                        +---images(image files)
#  thereby it is making sense to use torchvision.datasets.folder.ImageFolder to load the dataset
#   and we should transform the images to the same shape (3, 224, 224), then augment the source domian images
class MultiDomianImageFolderDataset(MultiDomainDataset):
    def __init__(self, root, no_of_test_domains, aug_transform, hparams):
        super().__init__()

        environments = [f.name for f in os.scandir(root) if f.is_dir()]
        environments = sorted(environments) # for example, PACS will get ['art_painting', 'cartoon', 'photo', 'sketch']
                                            # then following the order of DOMAINS which is ['A', 'C', 'P', 'S']

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        aug_transform = transforms.Compose([
            # transforms.Resize((224,224)),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.dataset = []
        # load the dataset, and augment the source domain images
        for i, foldername in enumerate(environments):
            path = os.path.join(root, foldername)

            if aug_transform and i not in no_of_test_domains:
                env_transform = aug_transform
            else:
                env_transform = transform

            env_dataset = torchvision.datasets.folder.ImageFolder(path, transform=env_transform)
            self.dataset.append(env_dataset)
        
        self.input_shape = (3,224,224)
        self.num_classes = len(self.dataset[-1].classes)

class PACSDataset(MultiDomianImageFolderDataset):
    STEPS = 15000
    CHECKPOINT_FREQ = 300
    DOMAINS = ['A', 'C', 'P', 'S']
    INPUT_SHAPE = (3, 224, 224)
    
    def __init__(self, root, no_of_test_domains, hparams):
        self.dir = os.path.join(root, 'PACS/')
        super().__init__(self.dir,no_of_test_domains,hparams['aug_transform'],hparams)
