from utils import *


def block2_standardize(
        input_data: Path, dict_out: dict, mean: list = None, std: list = None
    ) -> Union[tuple, ImageFolder]:

    normalized_photos_folder = dict_out['normalized_photos_folder']
    # resized_photos_folder = dict_out['resized_photos_folder']
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

    # if resized_dataset is not None:
    #     r_flag = True
    #     creates_folders(resized_dataset.classes, resized_photos_folder)

    #     start = time()
    #     save_image(resized_dataset, resized_photos_folder)
    #     print(f'({resized_photos_folder.name}) wasted time = {round(time() - start)} sec')

    if n_flag and r_flag:
        return normalized_dataset, resized_dataset
    elif n_flag:
        return normalized_dataset
    elif r_flag:
        return resized_dataset
    else:
        raise ValueError('No one of datasets wasn\'t created!')


def main():
    mean, std = ([0.516, 0.419, 0.373], [0.261, 0.235, 0.222])
    normalized_photos_folder = Path('data/tmp/standardized_with_normalized_105_pins')

    standard_dataset = block2_standardize(
        Path('data/input/105_classes_pins_dataset'),
        {'normalized_photos_folder': normalized_photos_folder},
        mean, std
    )


if __name__ == '__main__':
    main()
