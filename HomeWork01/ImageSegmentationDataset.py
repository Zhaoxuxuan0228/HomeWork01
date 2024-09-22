import os
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, mask_transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.mask_transform = mask_transform
        self.image_files = os.listdir(image_dir)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.image_files[idx])

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        return image, mask

    # 转换（可以根据需要进行修改）


transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
])

mask_transform = T.Compose([
    T.Resize((256, 256)),  # 确保mask和image尺寸相同
    T.ToTensor(),  # ToTensor会将PIL Image或NumPy ndarray转换为FloatTensor
])

# 创建数据集和数据加载器
train_dataset = CustomDataset(image_dir='archive/images', mask_dir='archive/labels', transform=transform,
                              mask_transform=mask_transform)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)