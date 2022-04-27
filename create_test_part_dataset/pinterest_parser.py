import time
import numpy as np
import argparse
import sys

from webdriver_manager.chrome import ChromeDriverManager

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

from typing import Tuple
from pathlib import Path
from tqdm import tqdm
from cv2 import imwrite, imdecode, resize
from urllib.request import urlopen

sys.path.append(f'{sys.path[0]}/..')
from utils import parallelize


parsed_photos = Path('data/parsed_from_pinterest')


def get_data(url: str, nmb: int) -> np.ndarray:
    options = Options()
    options.add_argument("--start-maximized")
    options.add_argument("--headless")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.get(url)

    times = 0
    q = 0
    bfn = 0
    samen = 0
    while True:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        if len(driver.find_elements(By.TAG_NAME, 'img')) >= nmb:
            break

        if len(driver.find_elements(By.TAG_NAME, 'img')) <= 5:
            q += 1
            if q == 50:
                print(f"Error! I can't find enough image!(Limit is 5, and there is no 5 image in this page! ) ")
                break

        if bfn == len(driver.find_elements(By.TAG_NAME, 'img')):
            samen += 1
            if samen == 500:
                print(f"There is no enough image, breaking down... I'm gonna download {bfn} image")
                nmb = bfn
                break
        times += 1
        bfn = len(driver.find_elements(By.TAG_NAME, 'img'))

        time.sleep(0.5)

    images = driver.find_elements(By.TAG_NAME, 'img')

    print(f'\n======> Was scrolled {times} times and find {len(images)} images!')

    assert len(images) > 4, \
        f'########### ERROR: Too small count of parsed images = {len(images)}. Need at least 5 ###########'

    res = []
    for idx, image in enumerate(images):
        if idx >= nmb:
            break
        res.append((image.get_attribute('src'), str(idx)))

    driver.close()

    assert len(res) == nmb, \
        f'########### ERROR: Incorrect len of parsed data, got {len(res)} but need {nmb} ###########'

    return np.array(res)


def downloads(data_ar: np.ndarray, folder_name: str, reshape: bool, shape: Tuple[int, int]):
    for data in data_ar.tolist():
        image = imdecode(np.asarray(bytearray(urlopen(data[0]).read()), dtype="uint8"), -1)

        if image is not None:
            if reshape:
                image = resize(image, shape)

            imwrite(f"{parsed_photos}/pins_{folder_name}/{folder_name}_{eval(data[1]):03d}.jpg", image)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument("--keyword", nargs='+', type=str, required=True,
                        help='List of keywords or all, example: --keyword "Adriana Lima" | example: --keyword all')

    parser.add_argument("--quantity", type=int, default=100, help='Count of images that need to download by keyword')

    parser.add_argument("--shape", nargs='+', type=int, default=[],
                        help='Need only 2 values (width, height) of image, example: --shape 256 256')

    return parser


def main() -> None:
    args = get_parser().parse_args()

    reshape = False
    new_shape = (256, 256)

    if args.shape:
        assert len(args.shape) == 2, \
            f'########### ERROR: Need only 2 values of shape images (width, height). But get: {args.input} ###########'
        reshape = True
        new_shape = args.shape

    if 'all' in args.keyword:
        keywords = sorted([p.name.split('_')[-1] for p in Path('data/input/105_classes_pins_dataset').glob('*')])
    else:
        keywords = args.keyword

    excess_keywords = list()
    if parsed_photos.is_dir():
        excess_keywords = sorted([p.name.split('_')[-1] for p in parsed_photos.glob('*')])

    tq_batch = tqdm(keywords, total=len(keywords))
    for keyword in tq_batch:
        if keyword in excess_keywords:
            continue

        tq_batch.set_description(f'Parsing...[{keyword}]')

        Path(f'{parsed_photos}/pins_{keyword}').mkdir(parents=True, exist_ok=True)
        link = f'https://tr.pinterest.com/search/pins/?q={keyword}'

        try:
            start = time.time()
            data_array = get_data(link, args.quantity)
            print(f'=========> Array which contains url and number of images for {keyword} was done !'
                  f'(wasted time = {time.time() - start:.2f} sec) <=========')

            parallelize(downloads, data_array, {'folder_name': keyword, 'reshape': reshape, 'shape': new_shape})
            print(f'=========> Parsing {keyword} is done!! <=========\n')
        except Exception as error:
            with open('data/errors_in_parsing.txt', mode='a') as f:
                print(f'{keyword}: {error}\n', file=f)
            print(f'########### ERROR: Problems with parsing {keyword}! ###########')
            continue


if __name__ == '__main__':
    main()
