import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
import face_recognition as fr

from PIL import Image, ImageDraw
# from mpl_toolkits.axes_grid1 import ImageGrid
from torchvision.datasets import ImageFolder

from functools import partial
from multiprocessing import cpu_count, Pool, current_process

from typing import Union

from gc import collect
from time import time
from tqdm import tqdm
from pathlib import Path


def get_photos(data: list, flag: bool = True) -> list:
    photos = []

    for p in data:
        image = Image.open(p.as_posix()).convert('RGB')
        image = image.resize((256, 256))
        photos.append(np.asarray(image))

    del image
    collect()

    return photos


def show_photos(data: list, grid_size: tuple):
    h, w = grid_size
    n = 0
    fig, ax = plt.subplots(h, w, figsize=(15, 15))
    fig.subplots_adjust(hspace=0, wspace=0)
    for i in range(h):
       for j in range(w):
          ax[i, j].xaxis.set_major_locator(plt.NullLocator())
          ax[i, j].yaxis.set_major_locator(plt.NullLocator())
          ax[i, j].imshow(data[n])
          n += 1
    plt.show()


def parallelize_get_statistics(func, data, kwargs={}, n_jobs=None) -> tuple:
    if n_jobs is None:
        cores = cpu_count()
    else:
        cores = n_jobs
    available_data = data.shape[0]
    if available_data < cores:
        cores = available_data
    data_split = np.array_split(data, cores)
    pool = Pool(cores)
    data = pool.map(partial(func, **kwargs), data_split)
    pool.close()
    pool.join()

    mean, std = np.mean(data, axis=0, dtype='float64')
    return [round(i, 3) for i in mean], [round(i, 3) for i in std]


def parallelize_filter_images(func, data, kwargs={}, n_jobs=None):
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


def get_statistics(paths: np.array) -> tuple:
    means, stds = [], []
    for p in tqdm(paths, desc=f'Progress bar ({current_process().name})...'):
        image = Image.open(p.as_posix())
        pixels = np.asarray(image, dtype='float64') / 255.0

        means.append(pixels.mean(axis=(0, 1), dtype='float64'))
        stds.append(pixels.std(axis=(0, 1), dtype='float64'))

    del pixels
    collect()

    return np.mean(means, axis=0, dtype='float64'), np.mean(stds, axis=0, dtype='float64')


def get_numpy_from_tenzor(image: torch.Tensor) -> np.ndarray:
    image = image.numpy().transpose(1, 2, 0)
    return np.clip(image, 0, 1)


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


def generate_mask(model: str = 'hog'):
    image = fr.load_image_file(
        'data/input/filtered_standardized_with_normalized_105_pins/pins_Alvaro Morte/Alvaro Morte19_212.jpg')
    face_landmarks_list = fr.face_landmarks(
        image,
        face_locations=fr.face_locations(image, number_of_times_to_upsample=0, model=model),
        model='custom'
    )
    print(face_landmarks_list)

    pil_image = Image.fromarray(image)
    d = ImageDraw.Draw(pil_image)

    d.polygon((face_landmarks_list[0]['mask_55']), fill="green")
    pil_image.show()


def filter_bad_images(p_list: list, out: Path, model: str = 'hog'):
    for i in tqdm(p_list, desc=f'Progress bar filtering ({current_process().name})...'):
        if i.is_file():
            image = fr.load_image_file(i.as_posix())
            if fr.face_landmarks(
                    image,
                    face_locations=fr.face_locations(image, number_of_times_to_upsample=0, model=model),
                    model='custom'
            ):
                Image.fromarray(image).save(f'{out}/{i.parts[-2]}/{i.name}')


def block1_get_stats(pict_list: list) -> tuple:
    # get statistics
    start = time()
    mean, std = parallelize_get_statistics(get_statistics, np.array(pict_list))
    print(f'wasted time = {round(time() - start)} sec')
    print(f'\nmean = {mean}\nstd = {std}')

    return mean, std


def block2_standardize(
        input_data: Path, dict_out: dict, mean: list = None, std: list = None) -> Union[tuple, ImageFolder]:

    normalized_photos_folder = dict_out['normalized_photos_folder']
    resized_photos_folder = dict_out['resized_photos_folder']
    normalized_dataset, resized_dataset = None, None
    n_flag, r_flag = False, False

    normalized_dataset = standardization(input_data, mean, std)
    # resized_dataset = standardization(input_data, flag=False)
    # print(len(normalized_dataset), normalized_dataset[0][0].size())  # 17534 torch.Size([3, 256, 256])

    if normalized_dataset is not None:
        n_flag = True
        # creates folders for standardised photos
        creates_folders(normalized_dataset.classes, normalized_photos_folder)

        # save images
        start = time()
        save_image(normalized_dataset, normalized_photos_folder)
        print(f'({normalized_photos_folder.name}) wasted time = {round(time() - start)} sec')

    if resized_dataset is not None:
        r_flag = True
        creates_folders(resized_dataset.classes, resized_photos_folder)

        start = time()
        save_image(resized_dataset, resized_photos_folder)
        print(f'({resized_photos_folder.name}) wasted time = {round(time() - start)} sec')

    if n_flag and r_flag:
        return normalized_dataset, resized_dataset
    elif n_flag:
        return normalized_dataset
    elif r_flag:
        return resized_dataset
    else:
        raise ValueError('No one of datasets wasn\'t created!')


def block3_filter(input_path: Path, output: Path, labels: list):
    creates_folders(labels, output)

    before_filter = list(input_path.rglob('*/*'))

    parallelize_filter_images(filter_bad_images, np.array(before_filter), {'out': output, 'model': 'cnn'})
    # filter_bad_images(before_filter, output, model='cnn')

    after_filter = len(list(output.rglob('*/*')))
    print(f'Count of deleted files = {len(before_filter) - after_filter}')  # old = 3445 | add face_locations = 3053


def main():
    pictures_folder = Path('data/input/105_classes_pins_dataset')

    pict_list = list(pictures_folder.rglob('*/*'))

    # before standardization photos (for example 64)
    # show_photos(get_photos(pict_list[:64]), (8, 8))

    ################################################################################

    # mean, std = block1_get_stats(pict_list)

    ################################################################################

    mean = [0.516, 0.419, 0.373]
    std = [0.261, 0.235, 0.222]

    normalized_photos_folder = Path('data/input/standardized_with_normalized_105_pins')
    resized_photos_folder = Path('data/input/standardized_with_resized_105_pins')

    # standard_dataset = block2_standardize(
    #     pictures_folder,
    #     {'normalized_photos_folder': normalized_photos_folder, 'resized_photos_folder': resized_photos_folder},
    #     mean, std
    # )
    #
    # if type(standard_dataset) == tuple:
    #     norm_dataset, resize_dataset = standard_dataset
    #
    #     del standard_dataset
    #     collect()

    normalized_dataset = standardization(pictures_folder, mean, std)

    ################################################################################

    filtered_normalized_photos_folder = Path('data/input/cnn_filtered_standardized_with_normalized_105_pins')

    block3_filter(
        input_path=normalized_photos_folder,
        output=filtered_normalized_photos_folder,
        labels=normalized_dataset.classes
    )

    ################################################################################

    generate_mask()





if __name__ == '__main__':
    main()
