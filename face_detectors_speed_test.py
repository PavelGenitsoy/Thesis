import face_recognition as fr

from mtcnn.mtcnn import MTCNN
from time import time
from tqdm import tqdm
from pathlib import Path


def worker(images: list, type_model: str):
    detector = MTCNN()
    count_skipped = 0
    for i in tqdm(images):
        try:
            if type_model == 'mtcnn':
                _ = detector.detect_faces(i)
            else:
                _ = fr.face_locations(i, 1, type_model)
        except:
            count_skipped += 1
            pass
    return count_skipped


def main():
    path_to_images = list(Path('data/input/105_classes_pins_dataset').glob('*/*'))

    images = [fr.load_image_file(i) for i in path_to_images[:50]]

    for model in ('hog', 'cnn', 'mtcnn'):
        start = time()
        skipped = worker(images, model)
        print(f'{model} processed by {time() - start:.1f} sec || And was skipped {skipped} images')


if __name__ == '__main__':
    main()
