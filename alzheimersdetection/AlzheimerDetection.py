import torch

'''
class AlzheimerResNet(nn.Module):
    def __init__(self, num_classes=4, pretrained=True):
        self.pretrained = pretrained
        self.model = resnet18(pretrained=pretrained)
        self.features = nn.Linear(self.model.fc.in_features, num_classes)
'''

class AlzheimerDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        if self.transform:
            image = self.transform(image)  # Apply transformations
        label = self.labels[index]
        return image, label