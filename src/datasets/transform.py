import torchvision.transforms as transforms

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
def without_augment(size=84, enlarge=False):
    if enlarge:
        resize = int(size*256./224.)
    else:
        resize = size
    return transforms.Compose([
                transforms.Resize(resize),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                normalize,
            ])

def with_augment(size=84, disable_random_resize=False):
    if disable_random_resize:
        return transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.RandomResizedCrop(size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])