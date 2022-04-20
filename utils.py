import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np
import torch
import boto3
import zipfile
import subprocess as sp

from PIL import Image

from sklearn.model_selection import train_test_split, KFold

# from sagemaker.utils import download_folder
# from sagemaker.session import Session

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, SubsetRandomSampler
# from torch.utils.data.dataset import Subset

from multiprocessing import cpu_count
from tqdm import tqdm
from typing import Tuple, List
from pathlib import Path
from time import time
from gc import collect


torch.manual_seed(47)

N_FOLDS = 5
etalon_datasets = {'standardized_with_normalized_105_pins': 'input/etalon_105_classes_pins_dataset.zip'}


def get_dict_from_list_of_dicts(l: List[dict]) -> dict:
    accuracy = [res['acc'] for res in l]
    losses = [res['loss'] for res in l]
    val_accuracy = [res['val_acc'] for res in l]
    val_losses = [res['val_loss'] for res in l]
    return {'accuracy': accuracy,
            'losses': losses,
            'val_accuracy': val_accuracy,
            'val_losses': val_losses}


def get_dls(
        ds: ImageFolder, train_idx: np.ndarray, valid_idx: np.ndarray, batch_size: int
) -> Tuple[DataLoader, DataLoader]:

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    n_cpu = cpu_count()

    dl = DataLoader(ds, batch_size=batch_size, num_workers=n_cpu, sampler=train_sampler)
    val_dl = DataLoader(ds, batch_size=batch_size if batch_size <= 16 else 16, num_workers=n_cpu, sampler=valid_sampler)
    return dl, val_dl


def split_ds(ds: ImageFolder) -> Tuple[np.ndarray, np.ndarray]:
    train_idx, test_idx = train_test_split(
        np.arange(len(ds)),
        test_size=0.1,
        shuffle=True,
        stratify=ds.targets,
        random_state=47
    )
    return train_idx, test_idx


def standardization(
        picts: Path, mean: list = None, std: list = None, size: int = 256, flag: bool = True
) -> ImageFolder:
    if flag:
        transformer = T.Compose([
            T.Resize(size),
            T.CenterCrop(size),
            T.ToTensor(),
            T.Normalize(*(mean, std))
        ])
    else:
        transformer = T.Compose([
            T.Resize(size),
            T.CenterCrop(size),
            T.ToTensor()
        ])

    return ImageFolder(picts.as_posix(), transform=transformer)


def show_photos(data: list, grid_size: tuple):
    data_ = []
    for ar in data:
        im_pillow = Image.fromarray((ar * 255).astype(np.uint8))
        im_pillow = im_pillow.resize((256, 256))
        data_.append(np.asarray(im_pillow))

    h, w = grid_size
    n = 0
    fig, ax = plt.subplots(h, w, figsize=(15, 15))
    fig.subplots_adjust(hspace=0, wspace=0)
    for i in range(h):
        for j in range(w):
            ax[i, j].xaxis.set_major_locator(plt.NullLocator())
            ax[i, j].yaxis.set_major_locator(plt.NullLocator())
            ax[i, j].imshow(data_[n])
            n += 1
    plt.show()


def get_photos(data: list, flag: bool = True) -> List[np.ndarray]:
    photos = []

    for p in data:
        image = Image.open(p.as_posix()).convert('RGB')
        image = image.resize((256, 256))
        photos.append(np.asarray(image))

    del image
    collect()

    return photos


def show_results(history: dict, model_folder: Path):
    (model_folder / Path('images')).mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.plot(history['losses'], '-o', label='Loss')
    ax1.plot(history['val_losses'], '-o', label='Validation Loss')
    ax1.legend()

    ax2.plot(100 * np.array(history['accuracy']), '-o', label='Accuracy')
    ax2.plot(100 * np.array(history['val_accuracy']), '-o', label='Validation Accuracy')
    ax2.legend()

    plt.savefig(f'{model_folder}/images/{model_folder.name}.png', bbox_inches='tight')
    # fig.show()


def get_numpy_from_tenzor(image: torch.Tensor) -> np.ndarray:
    image = image.numpy().transpose(1, 2, 0)
    return np.clip(image, 0, 1)


def download_and_unzip_dataset(
        bucket_input: str, bucket_output: str, s3_path_input: Path, s3_path_output: str, local_folder: Path
):
    try:
        print(f'Downloading dataset (zip archive) from s3 ({s3_path_input})')
        start = time()
        boto3.client('s3').download_file(bucket_input, s3_path_input.as_posix(), f'{local_folder}/{s3_path_input.name}')
        # download_folder(bucket, key, local_folder.as_posix(), Session())
        print(f'Dataset is downloaded! Wasted time = {time() - start:.1f} sec')
    except Exception as e:
        boto3.resource('s3').Object(
            bucket_output,
            f'{s3_path_output}/errors/failed_downloading.txt'
        ).put(Body=str(e))
        print(f'Report of error in {s3_path_output}/errors/failed_downloading.txt')
    try:
        tmp = list(local_folder.glob('*.zip'))
        for zip_file in tqdm(tmp, total=len(tmp), desc='Unzipping...'):
            with zipfile.ZipFile(zip_file.as_posix(), 'r') as zip_ref:
                zip_ref.extractall(local_folder.as_posix())
    except Exception as e:
        boto3.resource('s3').Object(
            bucket_output,
            f'{s3_path_output}/errors/failed_unzip.txt'
        ).put(Body=str(e))
        print(f'Report of error in {s3_path_output}/errors/failed_unzip.txt')


def upload_dire_to_s3(bucket_name: str, s3_path_output: str, local_path_to_folder: Path):
    print(f'Starting uploading results to s3 folder ({s3_path_output})')
    for item in local_path_to_folder.rglob('*'):
        if item.is_file():
            with open(item.as_posix(), 'rb') as data:
                s3_prefix = f'{s3_path_output}/models/{local_path_to_folder.parts[-1]}/' \
                            f'{item.as_posix()[len(local_path_to_folder.as_posix()) + 1:]}'
                boto3.resource('s3').Bucket(bucket_name).put_object(Key=s3_prefix, Body=data)
    print(f'Results from {local_path_to_folder} are uploaded to s3')


def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.used --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values[0]


def get_normal_photos(images_idxs: torch.Tensor, main_ds: ImageFolder) -> List[np.ndarray]:
    return [get_numpy_from_tenzor(main_ds[idx][0]) for idx in images_idxs.numpy()]


def rewrite_lib_file(path_to_file: str):
    with open(path_to_file, 'r') as f:
        lines = f.readlines()

    res = lines[:183] + ['        return sample, target, index\n'] + lines[184:]

    with open(path_to_file, 'w') as f:
        for item in res:
            f.write(item)


def cross_validation(dataset_indexes: np.ndarray, k_folds: int = 5):
    return KFold(n_splits=k_folds, shuffle=True).split(dataset_indexes)


def add_mean_std_to_list(l: List[float]):
    assert len(l) == N_FOLDS, '########### ERROR: Wrong len of list when count mean and std #############'

    mean = np.mean(l)
    std = np.std(l)

    l.extend([mean, std])

    return l