from os import listdir
from os.path import join

from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Scale,RandomHorizontalFlip,RandomVerticalFlip,RandomRotation,Resize,ColorJitter
import skimage.io

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG','bmp'])


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def train_hr_transform(crop_size):
    return Compose([
        RandomCrop(crop_size),
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        ColorJitter(0.2,0.2,0.1,0.1),
        ToTensor(),
    ])


def train_lr_transform(crop_size, upscale_factor):
    return Compose([
        ToPILImage(),
        Scale(crop_size // upscale_factor, interpolation=Image.BICUBIC),
        ToTensor()
    ])


def display_transform():
    return Compose([
        ToPILImage(),
        Scale(400),
        CenterCrop(400),
        ToTensor()
    ])


class TrainDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, crop_size, upscale_factor):
        super(TrainDatasetFromFolder, self).__init__()
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]
        crop_size = calculate_valid_crop_size(crop_size, upscale_factor)
        self.hr_transform = train_hr_transform(crop_size)
        self.lr_transform = train_lr_transform(crop_size, upscale_factor)

    def __getitem__(self, index):
        hr_image = self.hr_transform(Image.open(self.image_filenames[index]))
        lr_image = self.lr_transform(hr_image)
        return lr_image, hr_image

    def __len__(self):
        return len(self.image_filenames)


class ValDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(ValDatasetFromFolder, self).__init__()
        self.upscale_factor = upscale_factor
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]

    def __getitem__(self, index):
        hr_image = Image.open(self.image_filenames[index])
        hr_imgg = skimage.io.imread(self.image_filenames[index])
        if len(hr_imgg.shape) !=3 :
            hr_image = hr_image.convert('RGB')
        w, h = hr_image.size
        crop_size1 = calculate_valid_crop_size(w, self.upscale_factor*2)
        crop_size2 = calculate_valid_crop_size(h, self.upscale_factor*2)

        lr_scale = Resize((crop_size2 // self.upscale_factor, crop_size1 // self.upscale_factor), interpolation=Image.BICUBIC)
        hr_scale = Resize((crop_size2 , crop_size1), interpolation=Image.BICUBIC)


        hr_image = CenterCrop((crop_size2, crop_size1))(hr_image)
        #hr_image = hr_scale(hr_image)

        lr_image = lr_scale(hr_image)
        hr_restore_img = hr_scale(lr_image)

        return ToTensor()(lr_image), ToTensor()(hr_restore_img), ToTensor()(hr_image)

    def __len__(self):
        return len(self.image_filenames)


class TestDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(TestDatasetFromFolder, self).__init__()
        self.upscale_factor = upscale_factor
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]

    def __getitem__(self, index):
        hr_imgg = skimage.io.imread(self.image_filenames[index])
        hr_image = Image.open(self.image_filenames[index])
        if len(hr_imgg.shape) !=3 :
            hr_image = hr_image.convert('RGB')
        w, h = hr_image.size
        crop_size1 = calculate_valid_crop_size(w, 4)
        crop_size2 = calculate_valid_crop_size(h, 4)

        lr_scale = Resize((crop_size2 // 4, crop_size1 // 4), interpolation=Image.BICUBIC)
        hr_scale = Resize((crop_size2, crop_size1), interpolation=Image.BICUBIC)

        hr_image = CenterCrop((crop_size2, crop_size1))(hr_image)
        lr_image = lr_scale(hr_image)
        hr_restore_img = hr_scale(lr_image)

        return ToTensor()(lr_image), ToTensor()(hr_restore_img), ToTensor()(hr_image)

    def __len__(self):
        return len(self.image_filenames)


class TestDatasetFromFolder2(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(TestDatasetFromFolder2, self).__init__()
        self.upscale_factor = upscale_factor
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]

    def __getitem__(self, index):
        lr_noiseimg1 = skimage.io.imread(self.image_filenames[index])
        lr_noiseimg = Image.open(self.image_filenames[index])
        if len(lr_noiseimg1.shape) !=3 :
            lr_noiseimg = lr_noiseimg.convert('RGB')
        w, h = lr_noiseimg.size

        hr_scale = Resize((4*h, 4*w), interpolation=Image.BICUBIC)


        hr_restore_img = hr_scale(lr_noiseimg)

        return ToTensor()(lr_noiseimg), ToTensor()(hr_restore_img)

    def __len__(self):
        return len(self.image_filenames)


class TestDatasetFromFolder3(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(TestDatasetFromFolder3, self).__init__()
        self.upscale_factor = upscale_factor
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]

    def __getitem__(self, index):
        hr_imgg = skimage.io.imread(self.image_filenames[index])
        hr_image = Image.open(self.image_filenames[index])
        if len(hr_imgg.shape) !=3 :
            hr_image = hr_image.convert('RGB')
        w, h = hr_image.size
        lr_scale = Resize((h//2, w//2), interpolation=Image.BICUBIC)
        lrimg = lr_scale(hr_image)

        hr_scale = Resize((h * 4, w * 4), interpolation=Image.BICUBIC)
        bicimg = hr_scale(lrimg)
        return ToTensor()(lrimg), ToTensor()(bicimg)

    def __len__(self):
        return len(self.image_filenames)



class TestDatasetFromFolder4(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(TestDatasetFromFolder4, self).__init__()
        self.upscale_factor = upscale_factor
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]

    def __getitem__(self, index):
        lrt = skimage.io.imread(self.image_filenames[index])
        lr = Image.open(self.image_filenames[index])
        if len(lrt.shape) !=3 :
            lr = lr.convert('RGB')
        w, h = lr.size


        return ToTensor()(lr),self.image_filenames[index].split('/')[-1]

    def __len__(self):
        return len(self.image_filenames)
