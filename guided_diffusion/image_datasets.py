import math
import random
import collections

from PIL import Image
import blobfile as bf
import mpi4py
mpi4py.rc.thread_level="single"

from mpi4py import MPI
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from .batch_samplers import BalancedBatchSampler
from . import logger


def load_data(
    *,
    dataset,
    data_dir,
    images_id_file,
    batch_size,
    image_size,
    class_cond=False,
    deterministic=False,
    random_crop=False,
    random_flip=True,
    random_rotate=False,
    balance=True,
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    :param random_rotate: if True, randomly rotate the images for augmentation.
    """
    if not data_dir and not images_id_file:
        raise ValueError("unspecified data directory")
    if data_dir:
        all_files = _list_image_files_recursively(data_dir)
    elif images_id_file:
        logger.log(f'{dataset}')
        logger.log(f'Balance: {balance}')
        if 'csv' in images_id_file:
            df = pd.read_csv(images_id_file, low_memory=False)
            logger.log(f'Number of images: {df.shape[0]}')
            if dataset == 'eyepacs':
                data_dir = '/mnt/qb/eyepacs/data_processed/images/'
                df['image_full_path'] = df['image_path'].apply(lambda x: os.path.join(data_dir, x))

                all_files = df['image_full_path'].to_list()
                labels = df['diagnosis_image_dr_level'].to_list()
                labels = [3 if l>2 else int(l) for l in labels]
                logger.log(f'{collections.Counter(labels)}')
            elif dataset == 'eyepacs++':
                data_dirs = {'eyepacs': '/mnt/qb/eyepacs/data_processed/images/',
                             'benitez': '/mnt/qb/datasets/STAGING/berens/diffusion_model_data_mix/benitez/',
                             'fgadr': '/mnt/qb/datasets/STAGING/berens/diffusion_model_data_mix/fgadr/'
                            }

                df['parent_dir'] = df['dataset'].apply(lambda x: data_dirs[x])
                df['image_full_path'] = df['parent_dir']+df['image_path']

                all_files = df['image_full_path'].to_list()
                labels = df['label'].to_list()
                logger.log(f'{collections.Counter(labels)}')
            elif dataset == 'oct':
                data_dir = '/mnt/qb/berens/users/iilanchezian63/data/kermani_oct/CellData/OCT_preprocessed'
                df['image_full_path'] = df['filename'].apply(lambda x: os.path.join(data_dir, x))
                label_to_class = {'normal': 0, 'cnv': 1, 'drusen': 2, 'dme': 3}
                df['class'] = df['label'].apply(lambda x: label_to_class(x))

                all_files = df['image_full_path'].to_list()
                labels = df['class'].to_list()
                logger.log(f'{collections.Counter(labels)}')
        else:
            all_files = pickle.load(images_id_file)
    classes = None
    if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.
        class_names = [bf.basename(path).split("_")[0] for path in all_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]
    dataset = ImageDataset(
        image_size,
        all_files,
        classes=classes,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        random_crop=random_crop,
        random_flip=random_flip,
        random_rotate=random_rotate
    )
    if balance:
        sampler = BalancedBatchSampler(dataset, labels=labels)
        shuffle = False
    else:
        sampler = None
        shuffle = True
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, num_workers=1, drop_last=True,
            sampler=sampler
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, num_workers=1, drop_last=True,
            sampler=sampler
        )

    while True:
        yield from loader


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class ImageDataset(Dataset):
    def __init__(
        self,
        resolution,
        image_paths,
        classes=None,
        shard=0,
        num_shards=1,
        random_crop=False,
        random_flip=True,
        random_rotate=False
    ):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.random_rotate = random_rotate

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")

        if self.random_crop:
            arr = random_crop_arr(pil_image, self.resolution)
        else:
            arr = center_crop_arr(pil_image, self.resolution)

        if self.random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]

        if self.random_rotate and random.random() < 0.5:
            pil_image = Image.fromarray(arr)
            rotated_image = random_rotate_arr(pil_image, -15, 15)
            arr = np.asarray(rotated_image)

        arr = arr.astype(np.float32) / 127.5 - 1

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        return np.transpose(arr, [2, 0, 1]), out_dict


def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]

def random_rotate_arr(image, min_angle, max_angle):
    random_angle = random.randrange(min_angle, max_angle)
    rotated_img = image.rotate(random_angle)
    return rotated_img
