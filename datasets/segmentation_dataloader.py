from torch.utils.data import DataLoader
from torchvision import transforms

from .ade20k import ADE20KSegmentation
from .pascal_voc import VOCSegmentation

DATASET_DICT = {
    'ade20k': ADE20KSegmentation,
    'pascal_voc': VOCSegmentation,
}


def get_dataLoader(dataset, mode, batch_size, num_threads, data_path, base_size, crop_size):
    input_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225])
    ])
    data_kwargs = {'transform': input_transforms, 'base_size': base_size, 'crop_size': crop_size}
    dataset_class = DATASET_DICT[dataset]

    if mode == 'train':
        trainset = dataset_class(root=data_path, split=mode, mode=mode, **data_kwargs)
        return DataLoader(trainset, batch_size=batch_size, drop_last=True, shuffle=True, num_workers=num_threads, pin_memory=True)
    
    elif mode == 'val':
        validset = dataset_class(root=data_path, split=mode, mode=mode, **data_kwargs)
        return DataLoader(validset, batch_size=batch_size, shuffle=False, num_workers=num_threads, pin_memory=False)
    
    elif mode == 'test':
        testset = dataset_class(root=data_path, split=mode, mode=mode, **data_kwargs)
        return DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_threads, pin_memory=False)

    else:
        raise NotImplementedError
