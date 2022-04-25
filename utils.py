import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np
import torch
import boto3
import zipfile
import subprocess as sp
import argparse
import face_recognition as fr

from PIL.Image import Image
from PIL import ImageDraw

from sklearn.model_selection import train_test_split, KFold

from sagemaker.s3 import parse_s3_url
# from sagemaker.utils import download_folder
# from sagemaker.session import Session

from torchvision.datasets import ImageFolder

from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.data.dataset import Subset, ConcatDataset

from functools import partial
from multiprocessing import cpu_count, Pool, current_process

from tqdm import tqdm
from typing import Tuple, List, Union, Callable
from pathlib import Path
from time import time
from gc import collect

RS = 47
torch.manual_seed(RS)
np.random.seed(RS)

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


def split_ds(len_ds: int, targets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    train_idx, test_idx = train_test_split(
        np.arange(len_ds), test_size=0.1, shuffle=True, stratify=targets, random_state=RS
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


def show_photos(data: list, grid_size: tuple, image_size: Tuple[int, int] = (256, 256)):
    data_ = []
    for ar in data:
        im_pillow = Image.fromarray((ar * 255).astype(np.uint8))
        im_pillow = im_pillow.resize(image_size)
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


def get_photos(data: list, image_size: Tuple[int, int] = (256, 256), flag: bool = True) -> List[np.ndarray]:
    photos = []

    for p in data:
        image = Image.open(p.as_posix()).convert('RGB')
        image = image.resize(image_size)
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


def download_and_unzip_dataset(bucket_output: str, s3_inputs: list, s3_path_output: str, local_folder: Path):
    try:
        start = time()
        for t in s3_inputs:
            bucket_input, s3_path_input = t
            print(f'Downloading dataset (zip archive) from s3 ({s3_path_input})')
            boto3.client('s3').download_file(
                bucket_input, s3_path_input.as_posix(), f'{local_folder}/{s3_path_input.name}'
            )
            # download_folder(bucket, key, local_folder.as_posix(), Session())
        print(f'Dataset/s is/are downloaded! Wasted time = {time() - start:.1f} sec')
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


def rewrite_lib_file_torch_dataset(path_to_file: str):
    # python3.8/dist-packages/torchvision/datasets/folder.py
    with open(path_to_file, 'r') as f:
        lines = f.readlines()

    res = lines[:183] + ['        return sample, target, index\n'] + lines[184:]

    with open(path_to_file, 'w') as f:
        for item in res:
            f.write(item)


def rewrite_lib_file_fr_api(path_to_file: str):
    # python3.8/dist-packages/face_recognition/api.py
    with open(path_to_file, 'r') as f:
        lines = f.readlines()

    add_part_to_api_file = [
        "    elif model == 'custom':\n",
        '        return [{\n',
        '            "mask_55": points[2:15] + points[31:36]\n',
        '        } for points in landmarks_as_tuples]\n'
    ]

    res = lines[:198] + add_part_to_api_file + lines[198:]

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


def get_modified_s3_input_paths(inputs: Union[argparse.Namespace, list]) -> List[Tuple[str, Path]]:
    content = []
    for i in inputs:
        bucket_input, s3_path_input = parse_s3_url(i)
        s3_path_input = Path(s3_path_input.rstrip('/'))
        content.append((bucket_input, s3_path_input))
    return content


def get_binomial_distribution(n: int, p: float, size: int) -> np.ndarray:
    return np.random.binomial(n=n, p=p, size=size)


def get_mixed_dataset(
        s3_inputs: List[Tuple[str, Path]], input_datasets_folder: Path
) -> Tuple[int, np.ndarray, ConcatDataset]:

    _, s3_path_input1 = s3_inputs[0]
    standard_dataset1 = ImageFolder(
        (input_datasets_folder / s3_path_input1.stem).as_posix(), transform=T.ToTensor()
    )

    _, s3_path_input2 = s3_inputs[1]
    standard_dataset2 = ImageFolder(
        (input_datasets_folder / s3_path_input2.stem).as_posix(), transform=T.ToTensor()
    )

    assert len(standard_dataset1.classes) == len(standard_dataset2.classes), \
        '########### ERROR: different count of classes in 2 datasets ###########'

    assert len(standard_dataset1) == len(standard_dataset2), \
        '########### ERROR: different shape of datasets ###########'

    binomial_distribution = get_binomial_distribution(n=1, p=0.5, size=len(standard_dataset1))
    all_indexes = np.arange(len(standard_dataset1))

    subset1 = Subset(standard_dataset1, all_indexes[binomial_distribution.astype(bool)])
    subset2 = Subset(standard_dataset2, all_indexes[~binomial_distribution.astype(bool)])

    return len(standard_dataset1.classes), binomial_distribution, ConcatDataset([subset1, subset2])


def get_targets_for_mixed_dataset(
        standard_dataset: ConcatDataset, binomial_distribution: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:

    lbl = np.array([standard_dataset[i][1] for i in range(len(standard_dataset))])
    targets = np.concatenate((lbl.reshape((-1, 1)), binomial_distribution.reshape((-1, 1))), axis=1)

    return np.unique(targets), targets


def parallelize_filter_images(func: Callable, data: np.ndarray, kwargs: dict = {}, n_jobs=None):
    if n_jobs is None:
        cores = cpu_count()
    else:
        cores = n_jobs
    available_data = data.shape[0]
    if available_data < cores:
        cores = available_data
    data_split = np.array_split(data, cores)
    pool = Pool(cores)
    pool.map(partial(func, **kwargs), data_split)
    pool.close()
    pool.join()


def creates_folders(folders_names: list, default_folder: Path):
    for i in folders_names:
        (default_folder / Path(i)).mkdir(parents=True, exist_ok=True)
    print('\nFolders created!!')


def save_image(dataset: ImageFolder, default_folder: Path):
    for idx, (p, lbl) in tqdm(enumerate(dataset.imgs), total=len(dataset.imgs), desc=f'Progress bar...'):
        arr = get_numpy_from_tenzor(dataset[idx][0])

        image = Image.fromarray((arr * 255).astype(np.uint8)).convert('RGB')
        image.save(f"{default_folder}/{dataset.classes[lbl]}/{Path(p).name}")

    del arr
    collect()
    print('\nStandardised images saved!!')


def generate_mask(path: Path, model: str = 'hog') -> Union[Image, Path]:
    image = fr.load_image_file(path.as_posix())

    face_landmarks_list = fr.face_landmarks(
        image,
        face_locations=fr.face_locations(image, number_of_times_to_upsample=0, model=model),
        model='custom'
    )

    if face_landmarks_list:
        pil_image = Image.fromarray(image)
        d = ImageDraw.Draw(pil_image)
        d.polygon((face_landmarks_list[0]['mask_55']), fill='#00bfff')
        return pil_image
    else:
        return path


def processing_photos(dataset: np.ndarray, output_photos_folder: Path, classes: list, type_model: str):
    tqdm_batch = tqdm(dataset, total=dataset.shape[0], desc=f'Processing ({current_process().name})...')
    for path_to_file, lbl in tqdm_batch:
        path_to_file = Path(path_to_file)
        file_name, label = path_to_file.name, Path(classes[eval(lbl)])

        (output_photos_folder / label).mkdir(parents=True, exist_ok=True)

        pil_image = generate_mask(path_to_file, model=type_model)
        if isinstance(pil_image, Path):
            with open(f'data/skipped_images_for_{type_model}_by_{current_process().name}.txt', 'a') as f:
                print(pil_image, file=f)
        else:
            pil_image.save(output_photos_folder / label / file_name)


def get_cross_val_from_s3(bucket: str, s3_input: Path, local_folder: Path) -> Path:
    response = boto3.client('s3').list_objects_v2(Bucket=bucket, Prefix=s3_input.as_posix())

    assert response.get('Contents'), \
        f'########### ERROR: Attribute error. Response={response} doesn\'t have Contents ###########'

    path_to_save = Path()

    for d in response['Contents']:
        if d['Key'].endswith('/'):
            continue

        key = Path(d['Key'])
        if key.stem == 'history_cross_validation':
            key_parts = key.parts

            path_to_save = local_folder / Path(key_parts[0])
            path_to_save.mkdir(parents=True, exist_ok=True)

            boto3.client('s3').download_file(
                bucket, key.as_posix(), f'{path_to_save}/{key.stem}_{key_parts[-2]}{key.suffix}'
            )
    print(f'From {s3_input}: history_cross_validation files are downloaded!')

    return path_to_save
