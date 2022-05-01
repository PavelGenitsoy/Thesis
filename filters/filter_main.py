import sys
import os

from PIL import Image

sys.path.append(f'{sys.path[0]}/..')
from utils import (creates_folders, parallelize, parallelize_upd, save_image, standardization)
from utils import (RS, MEAN, STD, List, Tuple)
from utils import (torch, np, Path, tqdm, fr, current_process, ImageFolder, time)


torch.manual_seed(RS)
np.random.seed(RS)


def filter_bad_images(p_list: list, out: Path, model: str):
    for i in tqdm(p_list, total=len(p_list), desc=f'Filtering images ({current_process().name})...'):
        if i.is_file():
            image = fr.load_image_file(i.as_posix())
            if fr.face_landmarks(
                    image,
                    face_locations=fr.face_locations(image, number_of_times_to_upsample=0, model=model),
                    model='custom'
            ):
                Image.fromarray(image).save(f'{out}/{i.parts[-2]}/{i.name}')


def get_and_save_faces_from_image(images: list, shape: Tuple[int, int], out: Path, model: str) -> List[Path]:
    bad_img = []
    for image_path in tqdm(images, total=len(images), desc=f'Extracting faces ({current_process().name})...'):
        if image_path.is_file():
            image = fr.load_image_file(image_path)
            face_locations = fr.face_locations(image, number_of_times_to_upsample=1, model=model)
            if face_locations:
                for idx, face_location in enumerate(face_locations):
                    top, right, bottom, left = face_location

                    img_arr = Image.fromarray(image[top:bottom, left:right]).resize(shape)
                    img_arr.save(f'{out}/{image_path.parent.name}/{image_path.stem}_{idx}{image_path.suffix}')
            else:
                bad_img.append(image_path)
    return bad_img


def launch_standardise(d: Path, folder: Path):
    norm_d = standardization(d, MEAN, STD)

    creates_folders(norm_d.classes, folder)

    start = time()
    save_image(norm_d, folder)
    print(f'({folder.name}) wasted time = {round(time() - start)} sec')


def launch_filter(input_path: Path, output: Path, labels: list, model: str):
    creates_folders(labels, output)

    before_filter = list(input_path.rglob('*/*'))
    parallelize(filter_bad_images, np.array(before_filter), {'out': output, 'model': model})
    count_after_filter = len(list(output.rglob("*/*")))
    print(f'Count of images before 1 stage filter faces is {len(before_filter)}\n'
          f'Count of images after 1 stage filter faces is {count_after_filter}\n'
          f'Was removed {len(before_filter) - count_after_filter}')


def launch_get_faces(root: str, input_path: Path, output: Path, labels: list, model: str, shape: Tuple[int, int]):
    creates_folders(labels, output)

    before = list(input_path.glob('*/*'))
    skipped_img = parallelize_upd(
        get_and_save_faces_from_image, np.array(before), {'shape': shape, 'out': output, 'model': model}
    )

    with open(f'{root}/not_found_faces_{model}.txt', 'a') as f:
        for item in skipped_img:
            print(item, file=f)

    count_after = len(list(output.glob("*/*")))
    print(
        f'Count of images before 1 stage extract faces is {len(before)}\n'
        f'Count of images after 1 stage extract faces is {count_after}'
    )


# standardise dataset and save
def stage_0_standardise(root_prefix: str):
    folder_with_all_photos = Path(f'{root_prefix}/parsed_from_pinterest')
    standard_folder_with_all_photos = Path(f'{root_prefix}/standard_parsed_from_pinterest')

    assert folder_with_all_photos.is_dir(), '########### ERROR: Input folder isn\'t exists! ###########'

    launch_standardise(folder_with_all_photos, standard_folder_with_all_photos)


# remove photos where not found any faces and save one face to one image
def stage_1_filter_and_faces(root_prefix: str, model: str, shape: Tuple[int, int] = (256, 256)):
    folder_with_all_photos = Path(f'{root_prefix}/parsed_from_pinterest')
    filter_folder_with_all_photos = Path(f'{root_prefix}/filter_{model}_parsed_from_pinterest')
    folder_with_only_1_face = Path(f'{root_prefix}/resized_one_face_filter_{model}_parsed_from_pinterest')

    assert folder_with_all_photos.is_dir(), '########### ERROR: Input folder isn\'t exists! ###########'

    labels = [p.name for p in folder_with_all_photos.glob('*')]

    # launch_filter(input_path=folder_with_all_photos, output=filter_folder_with_all_photos, labels=labels, model=model)

    assert filter_folder_with_all_photos.is_dir(), '########### ERROR: Input folder isn\'t exists! ###########'

    launch_get_faces(
        root=root_prefix,
        input_path=filter_folder_with_all_photos,
        output=folder_with_only_1_face,
        labels=labels,
        model=model,
        shape=shape
    )


def main():
    os.chdir(sys.path[-1])
    root = 'data/datasets_for_testing_part'

    # stage_0_standardise(root)
    stage_1_filter_and_faces(root, 'hog')
    # stage_1_filter_and_faces(root, 'cnn')


if __name__ == '__main__':
    main()
