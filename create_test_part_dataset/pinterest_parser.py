import time
import argparse

from pathlib import Path

from webdriver_manager.chrome import ChromeDriverManager

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

from tqdm import tqdm
from threading import Thread, Lock
from queue import Queue
from cv2 import imwrite, imdecode, resize
from urllib.request import urlopen
from numpy import asarray

lock = Lock()


class bingo:
    DONE = False


class Ilister:
    ilist = Queue()
    copy_l = []
    leni = 0
    fn = None

    def __init__(self, nmb: int, fn: str):
        Ilister.fn = fn
        options = Options()
        options.add_argument("--start-maximized")
        options.add_argument("--headless")
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
        link = f'https://tr.pinterest.com/search/pins/?q={fn}'
        driver.get(link)

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

            bfn = len(driver.find_elements(By.TAG_NAME, 'img'))

            time.sleep(0.5)

        images = driver.find_elements(By.TAG_NAME, 'img')

        Path(f'data/parsed_from_pinterest/pins_{fn}').mkdir(parents=True, exist_ok=True)

        q = 0
        for image in images:
            if q >= nmb:
                break
            item = (image.get_attribute('src').replace("236x", "564x"), q)
            Ilister.copy_l.append(item)
            Ilister.ilist.put(item)
            q += 1
        Ilister.leni = Ilister.ilist.qsize()

        driver.close()


class Downloader:
    time1 = None
    q = 0

    def __init__(self, reshapep: bool, shape: tuple, pbar, iq=20):
        self.fn = Ilister.fn
        self.reshapep = reshapep
        self.shape = shape
        self.pbar = pbar

        Downloader.time1 = time.time()
        Downloader.q = 0
        for i in range(5):
            t = Thread(target=self.checker)
            t.daemon = True
            t.start()

        self.fpi = (100 - iq) / int(Ilister.ilist.qsize())
        self.proc = iq
        Ilister.ilist.join()
        del Ilister.ilist
        Ilister.ilist = Queue()

    def downloads(self, image):
        number = image[1]
        image = image[0]
        img = imdecode(asarray(bytearray(urlopen(image).read()), dtype="uint8"), -1)

        if img is not None:
            if self.reshapep:
                img = resize(img, self.shape)

            imwrite(f"data/parsed_from_pinterest/pins_{self.fn}/{self.fn}{number}.jpg", img)

        self.proc += self.fpi
        if self.pbar is not None:
            self.pbar.setValue(int(self.proc))

        Downloader.q += 1

    def checker(self):
        while True:
            image = Ilister.ilist.get()
            self.downloads(image)
            Ilister.ilist.task_done()


def launching(keyword: str, numberl: int, reshape: bool, shape: tuple = (256, 256), pbar=None):
    if pbar is not None:
        pbar.setValue(5)

    Ilister(numberl, keyword)

    if Ilister.leni <= 5:
        print(f"I can't find any image about {keyword}")
    else:
        Downloader(reshape, shape, pbar)

    bingo.DONE = Thread


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

    reshaped = False
    new_shape = tuple()

    if args.shape:
        assert len(args.shape) == 2, \
            f'########### ERROR: Need only 2 values of shape images (width, height). But get: {args.input} ###########'
        reshaped = True
        new_shape = args.shape

    if 'all' in args.keyword:
        keywords = sorted([p.name.split('_')[-1] for p in Path('data/input/105_classes_pins_dataset').glob('*')])
    else:
        keywords = args.keyword

    tq_batch = tqdm(keywords, total=len(keywords))
    for keyword in tq_batch:
        tq_batch.set_description(f'Parsing...[{keyword}]')

        try:
            launching(keyword, args.quantity, reshaped, new_shape)
            print(f'=========> Parsing {keyword} is done!! <=========')
        except Exception as error:
            with open('data/errors_in_parsing.txt', mode='a') as f:
                print(f'{keyword}: {error}\n', file=f)
            print(f'########### ERROR: Problems with parsing {keyword}! ###########')
            continue


if __name__ == '__main__':
    main()
