import sys
import os
import json
import random
import shutil
import face_recognition as fr

from PIL import Image

sys.path.append(f'{sys.path[0]}/..')
from utils import (creates_folders, parallelize, parallelize_upd, save_image, standardization)
from utils import (RS, MEAN, STD, List, Tuple)
from utils import (torch, np, Path, tqdm, current_process, ImageFolder, time)


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


def find_incorrect_photos(f_list: list, root: Path, shape: Tuple[int, int]) -> None:
    d_main_results = dict()
    face_location = [(0, shape[0], shape[1], 0)]
    for folder_path in tqdm(f_list, total=len(f_list), desc=f'Processing folders ({current_process().name})...'):
        if not folder_path.is_dir():
            continue

        d_results = dict()
        path_to_files = list(folder_path.glob('*'))

        for idx_outer, image_path in enumerate(path_to_files):
            if not image_path.is_file():
                continue

            image_main = fr.face_encodings(fr.load_image_file(image_path), face_location, 3, 'large')[0]

            distances = []
            for idx_inner, path_to_file in enumerate(path_to_files):
                if not path_to_file.is_file():
                    continue

                if idx_outer != idx_inner:
                    encoding = fr.face_encodings(fr.load_image_file(path_to_file), face_location, 3, 'large')[0]
                    distance = fr.face_distance([encoding], image_main)[0]

                    if 0.23 < distance <= 0.25:
                        with open(f'{root}/similar_images_{current_process().name}.txt', 'a') as f:
                            print(f'{image_path.stem} similar with {path_to_file.stem}: {distance}', file=f)
                        continue
                    elif distance <= 0.23:
                        path_to_file.unlink()
                        try:
                            del d_results[path_to_file.name]
                        except:
                            pass
                        continue

                    distances.append(distance)

            d_results[image_path.name] = distances
        d_main_results[folder_path.as_posix()] = d_results
    with open(f'{root}/data_{current_process().name}.json', 'w') as fp:
        json.dump(d_main_results, fp)


def find_incorrect_photos_v2(f_list: list, root: Path, shape: Tuple[int, int]) -> None:
    d_main_results = dict()
    face_location = [(0, shape[0], shape[1], 0)]
    for folder_path in tqdm(f_list, total=len(f_list), desc=f'Processing folders ({current_process().name})...'):
        if not folder_path.is_dir():
            continue

        d_results = dict()
        path_to_files = list(folder_path.glob('*'))

        etalon_prefix = Path(f'data/datasets_for_testing_part/10_images_by_class_valid/{folder_path.name}')
        etalon_images = list(etalon_prefix.glob('*'))

        etalon_encodings = \
            [fr.face_encodings(fr.load_image_file(path), face_location, 3, 'large')[0] for path in etalon_images]

        for idx_outer, image_path in enumerate(path_to_files):
            if image_path.name in list(map(lambda x: x.name, etalon_images)):
                continue

            encoding = fr.face_encodings(fr.load_image_file(image_path), face_location, 3, 'large')[0]
            distance = fr.face_distance(etalon_encodings, encoding)

            checks_etalon = [True if val <= 0.25 else False for val in list(distance)]
            if any(checks_etalon):
                with open(f'{root}/VERY_similar_images_{current_process().name}.txt', 'a') as f:
                    print(f'{image_path.stem} very similar with {checks_etalon.count(True)} of etalon images: '
                          f'{np.array([i.name for i in etalon_images])[checks_etalon]}', file=f)

            distance = np.mean(distance)

            if 0.23 < distance <= 0.25:
                with open(f'{root}/similar_images_{current_process().name}.txt', 'a') as f:
                    print(f'{image_path.stem} similar with 10 etalon images: {distance}', file=f)
                continue
            # elif distance <= 0.23:
            #     image_path.unlink()
            #     try:
            #         del d_results[path_to_file.name]
            #     except:
            #         pass
            #     with open(f'{root}/VERY_similar_images_{current_process().name}.txt', 'a') as f:
            #         print(f'{image_path.stem} similar with {image_path.stem}: {distance}', file=f)
            #     continue

            d_results[image_path.name] = distance
        d_main_results[folder_path.as_posix()] = d_results
    with open(f'{root}/data_{current_process().name}.json', 'w') as fp:
        json.dump(d_main_results, fp)


def find_duplicate_in_records(data: List[set]) -> dict:
    d_duplicates = dict()
    for idx, pair in enumerate(data):
        mask_duplicates = [pair == tmp if idx != idx_inner else False for idx_inner, tmp in enumerate(data)]
        index_of_duplicate = np.where(mask_duplicates)[0]
        if index_of_duplicate.size:
            pair = list(pair)
            if pair[0] in d_duplicates.keys() or pair[1] in d_duplicates.keys():
                continue
            d_duplicates[pair[0]] = index_of_duplicate[0]
    return d_duplicates


# standardise dataset and save
def stage_3_standardise(root_prefix: str):
    folder_with_all_photos = Path(f'{root_prefix}/all_unique_parsed_images')
    standard_folder_with_all_photos = Path(f'{root_prefix}/standard_all_unique_parsed_images')

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


def stage_2_find_duplicates(root_prefix: Path, shape: Tuple[int, int] = (256, 256)):
    # p = Path('data/datasets_for_testing_part/resized_one_face_filter_hog_parsed_from_pinterest')
    p = Path('data/datasets_for_testing_part/after_duplicates_resized_one_face_filter_hog_parsed_from_pinterest')
    root_prefix = root_prefix / Path('logs_compare_etalon')
    root_prefix.mkdir(parents=True, exist_ok=True)

    before = list(p.glob('*'))
    # parallelize(find_incorrect_photos, np.array(before), {'shape': shape, 'root': root_prefix})
    parallelize(find_incorrect_photos_v2, np.array(before), {'shape': shape, 'root': root_prefix})

    with open(f'{root_prefix}/similar_images.txt', 'w') as outfile:
        for filename in root_prefix.glob('*.txt'):
            with open(filename) as infile:
                outfile.write(infile.read())

    with open(f'{root_prefix}/similar_images_upd.txt', 'w') as f_out:
        with open(f'{root_prefix}/similar_images.txt') as f_in:
            main_content = f_in.readlines()
            tmp = []
            content = list(map(lambda x: x.rstrip().split(':')[0], main_content.copy()))
            for line in content:
                first_part = line.split('similar with')[0].rstrip()
                second_part = line.split('similar with')[1].lstrip()
                tmp.append({first_part, second_part})

            for i in sorted(find_duplicate_in_records(tmp).values(), reverse=True):
                print(f'Deleting record ==> {main_content[i]}')
                main_content.pop(i)
            f_out.write(''.join(main_content))

    data = dict()
    for filename in root_prefix.glob('*.json'):
        with open(filename) as infile:
            data.update(json.load(infile))
    with open(f'{root_prefix}/data.json', 'w') as outfile:
        json.dump(data, outfile, sort_keys=True)


def test(root_prefix: str, model: str, shape: Tuple[int, int] = (256, 256)):
    f = Path('data/datasets_for_testing_part/manual_from_google/pins_Amanda Crew')
    o = Path('data/datasets_for_testing_part/manual_from_google/filtered')

    creates_folders([f.name], o)

    before = list(f.glob('*'))

    skipped_img = parallelize_upd(
        get_and_save_faces_from_image, np.array(before), {'shape': shape, 'out': o, 'model': model}
    )

    with open(f'{root_prefix}/not_found_faces_{model}_test.txt', 'a') as f:
        for item in skipped_img:
            print(item, file=f)

    count_after = len(list(o.glob("*/*")))
    print(
        f'Count of images before 1 stage extract faces is {len(before)}\n'
        f'Count of images after 1 stage extract faces is {count_after}'
    )


def resize_img(input_folder: list, shape: Tuple[int, int], root: Path):
    for f in tqdm(input_folder, total=len(input_folder), desc=f'Processing folders ({current_process().name})...'):
        key = Path(f'{root.parent}/resized_{root.name}/{f.name}')
        key.mkdir(parents=True, exist_ok=True)
        for p in f.glob('*'):
            image = Image.open(p).resize(shape)
            image.save(key / Path(p.name))
    print('Resizing is done!')


def launch_resize(input_folder: Path, shape: Tuple[int, int]):
    l = list(input_folder.glob('*'))
    parallelize(resize_img, np.array(l), {'shape': shape, 'root': input_folder})


def select_images(input_folder: list, root: Path, folder_suitable_img: Path):
    for f in tqdm(input_folder, total=len(input_folder), desc=f'Processing folders ({current_process().name})...'):
        key = Path(f'{root}/{f.name}')
        key.mkdir(parents=True, exist_ok=True)
        for p in random.sample(list((folder_suitable_img / Path(f.name)).glob('*')), 25):
            shutil.copy(f / Path(p.name), key / Path(p.name))
    print('Selecting images are done!')


def launch_selecting_images(input_folder: Path):
    folder_suitable_img = Path('data/datasets_for_testing_part/mask_cnn_standard_all_unique_parsed_images')
    o = Path('data/datasets_for_testing_part/final_test_dataset')
    l = list(input_folder.glob('*'))
    parallelize(select_images, np.array(l), {'root': o, 'folder_suitable_img': folder_suitable_img})


def main():
    os.chdir(sys.path[-1])
    root = 'data/datasets_for_testing_part'

    # stage_0_standardise(root)
    # stage_1_filter_and_faces(root, 'hog')
    # stage_1_filter_and_faces(root, 'cnn')

    # test(root, 'hog')

    # stage_2_find_duplicates(Path(root))

    # launch_resize(Path('data/input/105_classes_pins_dataset'), (256, 256))

    # stage_3_standardise(root)

    launch_selecting_images(Path('data/datasets_for_testing_part/standard_all_unique_parsed_images'))


if __name__ == '__main__':
    main()
