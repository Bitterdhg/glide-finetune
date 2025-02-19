import time
from pathlib import Path
from random import randint, choice, random

import PIL

import torch as th
from torch.utils.data import Dataset
from torchvision import transforms as T
from glide_finetune.glide_util import get_tokens_and_mask, get_uncond_tokens_mask
from glide_finetune.train_util import pil_image_to_norm_tensor

from torch.utils import data
from torchvision import transforms, utils
from PIL import Image
import os
import numpy as np

def random_resized_crop(image, shape, resize_ratio=1.0):
    """
    Randomly resize and crop an image to a given size.

    Args:
        image (PIL.Image): The image to be resized and cropped.
        shape (tuple): The desired output shape.
        resize_ratio (float): The ratio to resize the image.
    """
    image_transform = T.RandomResizedCrop(shape, scale=(resize_ratio, 1.0), ratio=(1.0, 1.0))
    return image_transform(image)


def get_image_files_dict(base_path):
    image_files = [
        *base_path.glob("**/*.png"),
        *base_path.glob("**/*.jpg"),
        *base_path.glob("**/*.jpeg"),
        *base_path.glob("**/*.bmp"),
    ]
    return {image_file.stem: image_file for image_file in image_files}


def get_text_files_dict(base_path):
    text_files = [*base_path.glob("**/*.txt")]
    return {text_file.stem: text_file for text_file in text_files}


def get_shared_stems(image_files_dict, text_files_dict):
    image_files_stems = set(image_files_dict.keys())
    text_files_stems = set(text_files_dict.keys())
    return list(image_files_stems & text_files_stems)


# fscoco datasets
class TripleDataset(data.Dataset):
    def __init__(self, photo_root, sketch_root, text_root):
        super(TripleDataset, self).__init__()

        self.real_tranform = transforms.Compose([
            transforms.Resize([64, 64]),
            # transforms.CenterCrop(256),
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.sketch_tranform = transforms.Compose([
            transforms.Resize([224, 224]),
            # transforms.CenterCrop(256),
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        # tranform rgb to sketch
        # self.sketch_tranform = transforms.Compose([transforms.functional.rgb_to_grayscale])

        classes, class_to_idx = self.find_classes(photo_root)

        self.photo_root = photo_root
        self.sketch_root = sketch_root
        self.text_root = text_root

        self.photo_paths = sorted(self.make_dataset(self.photo_root))
        self.classes = classes
        self.class_to_idx = class_to_idx

        self.len = len(self.photo_paths)

    def __getitem__(self, index):

        photo_path = self.photo_paths[index]
        sketch_path, label, text = self._getrelate_sketch(photo_path)

        photo = Image.open(photo_path).convert('RGB')
        sketch = Image.open(sketch_path).convert('RGB')

        P = self.real_tranform(photo)
        S = self.sketch_tranform(sketch)
        # S = self.sketch_tranform(S) # tranform rgb to gray
        L = label
        T = text
        return {'P': P, 'S': S, 'L': L, 'T': T}

    def __len__(self):
        return self.len

    def make_dataset(self, root):
        images = []
        cnames = os.listdir(root)
        for cname in cnames:
            c_path = os.path.join(root, cname)
            if os.path.isdir(c_path):
                fnames = os.listdir(c_path)
                for fname in fnames:
                    path = os.path.join(c_path, fname)
                    images.append(path)
        return images

    def find_classes(self, root):
        classes = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
        classes.sort()
        class_to_idex = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idex

    def _getrelate_sketch(self, photo_path):

        paths = photo_path.split('/')
        fname = paths[-1].split('.')[0]
        cname = paths[-2]
        label = self.class_to_idx[cname]
        sketchs = sorted(os.listdir(os.path.join(self.sketch_root, cname)))
        sketch_rel = []
        for sketch_name in sketchs:
            if sketch_name.split('.')[0] == fname:
                sketch_rel.append(sketch_name)
        rnd = np.random.randint(0, len(sketch_rel))
        sketch = sketch_rel[rnd]
        # load text
        text_sigle = sketch.split('.')[0]+'.txt'
        sketch_path = os.path.join(self.sketch_root, cname, sketch)
        text_path = os.path.join(self.text_root, cname, text_sigle)
        f = open(text_path)
        text = f.read()
        f.close()
        return sketch_path, label, text



class TextImageDataset(Dataset):
    def __init__(
        self,
        folder="",
        side_x=64,
        side_y=64,
        resize_ratio=0.75,
        shuffle=False,
        tokenizer=None,
        text_ctx_len=128,
        uncond_p=0.0,
        use_captions=False,
        enable_glide_upsample=False,
        upscale_factor=4,
    ):
        super().__init__()
        folder = Path(folder)

        self.image_files = get_image_files_dict(folder)# {'000000281581': PosixPath('/root/Pro...1581.jpg'), '000000406874': Posi
        if use_captions:# True
            self.text_files = get_text_files_dict(folder)# {'000000197222': PosixPath('/root/Pro...7222.txt'),
            self.keys = get_shared_stems(self.image_files, self.text_files)
            print(f"Found {len(self.keys)} images.")# 100
            print(f"Using {len(self.text_files)} text files.")# 100
        else:
            self.text_files = None
            self.keys = list(self.image_files.keys())
            print(f"Found {len(self.keys)} images.")
            print(f"NOT using text files. Restart with --use_captions to enable...")
            time.sleep(3)

        self.resize_ratio = resize_ratio
        self.text_ctx_len = text_ctx_len

        self.shuffle = shuffle
        self.prefix = folder
        self.side_x = side_x
        self.side_y = side_y
        self.tokenizer = tokenizer
        self.uncond_p = uncond_p
        self.enable_upsample = enable_glide_upsample
        self.upscale_factor = upscale_factor

    def __len__(self):
        return len(self.keys)

    def random_sample(self):
        return self.__getitem__(randint(0, self.__len__() - 1))

    def sequential_sample(self, ind):
        if ind >= self.__len__() - 1:
            return self.__getitem__(0)
        return self.__getitem__(ind + 1)

    def skip_sample(self, ind):
        if self.shuffle:
            return self.random_sample()
        return self.sequential_sample(ind=ind)

    def get_caption(self, ind):
        # --------------------------------------------------------------------------------------------
        # step 1 : 得到idx text desceiption
        key = self.keys[ind]
        text_file = self.text_files[key]
        descriptions = open(text_file, "r").readlines()
        descriptions = list(filter(lambda t: len(t) > 0, descriptions))
        # --------------------------------------------------------------------------------------------

        # --------------------------------------------------------------------------------------------
        # step 2 : 得到token 和 mask
        try:
            description = choice(descriptions).strip()
            return get_tokens_and_mask(tokenizer=self.tokenizer, prompt=description)
        except IndexError as zero_captions_in_file_ex:
            print(f"An exception occurred trying to load file {text_file}.")
            print(f"Skipping index {ind}")
        # --------------------------------------------------------------------------------------------
            return self.skip_sample(ind)

    def __getitem__(self, ind):
        # ---------------------------------------------------------------------------------------------------
        # step 1 : 得到idx
        key = self.keys[ind]
        image_file = self.image_files[key]
        if self.text_files is None or random() < self.uncond_p:# self.uncond_p=0.2 cls_free
            tokens, mask = get_uncond_tokens_mask(self.tokenizer)
        else:
            tokens, mask = self.get_caption(ind)# tokens.shape([128]) mask.shape([128])
        # ---------------------------------------------------------------------------------------------------

        # ---------------------------------------------------------------------------------------------------
        # step 2 : 是否需要up
        try:
            original_pil_image = PIL.Image.open(image_file).convert("RGB")
        except (OSError, ValueError) as e:
            print(f"An exception occurred trying to load file {image_file}.")
            print(f"Skipping index {ind}")
            return self.skip_sample(ind)
        if self.enable_upsample: # the base image used should be derived from the cropped high-resolution image.
            upsample_pil_image = random_resized_crop(original_pil_image, (self.side_x * self.upscale_factor, self.side_y * self.upscale_factor), resize_ratio=self.resize_ratio)
            upsample_tensor = pil_image_to_norm_tensor(upsample_pil_image)
            base_pil_image = upsample_pil_image.resize((self.side_x, self.side_y), resample=PIL.Image.BICUBIC)
            base_tensor = pil_image_to_norm_tensor(base_pil_image)
            return th.tensor(tokens), th.tensor(mask, dtype=th.bool), base_tensor, upsample_tensor
        # ---------------------------------------------------------------------------------------------------


        # ---------------------------------------------------------------------------------------------------
        # step 3 : transforms
        base_pil_image = random_resized_crop(original_pil_image, (self.side_x, self.side_y), resize_ratio=self.resize_ratio)
        base_tensor = pil_image_to_norm_tensor(base_pil_image)# base_tensor.Size([3, 64, 64])
        # ---------------------------------------------------------------------------------------------------

        return th.tensor(tokens), th.tensor(mask, dtype=th.bool), base_tensor# base_tensor.Size([3, 64, 64])