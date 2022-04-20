import face_recognition as fr

from utils import *

from PIL import ImageDraw
from functools import partial
from multiprocessing import cpu_count, Pool, current_process

from typing import Callable, Union


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
