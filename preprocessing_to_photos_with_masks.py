from utils import *


def main():
    cnn_filtered_photos_folder = Path('data/input/cnn_filtered_standardized_with_normalized_105_pins')
    hog_filtered_photos_folder = Path('data/input/filtered_standardized_with_normalized_105_pins')

    mask_cnn_filtered_photos_folder = Path('data/input/masks_cnn_filtered_standardized_with_normalized_105_pins')
    mask_hog_filtered_photos_folder = Path('data/input/masks_hog_filtered_standardized_with_normalized_105_pins')

    triple = (
        ('cnn', cnn_filtered_photos_folder, mask_cnn_filtered_photos_folder),
        ('hog', hog_filtered_photos_folder, mask_hog_filtered_photos_folder)
    )

    tqdm_batch_inputs = tqdm(triple, total=len(triple))
    for type_model, input_photos_folder, output_photos_folder in tqdm_batch_inputs:
        filtered_dataset = ImageFolder(input_photos_folder.as_posix(), transform=transforms.ToTensor())

        tqdm_batch_inputs.set_description(f'Filtered by {type_model}')

        params = {
            'classes': filtered_dataset.classes,
            'output_photos_folder': output_photos_folder,
            'type_model': type_model
        }

        parallelize_filter_images(processing_photos, np.array(filtered_dataset.imgs), params)


if __name__ == '__main__':
    main()
