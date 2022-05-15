import argparse
import pandas as pd
import seaborn as sns
import json

from sklearn.metrics import confusion_matrix, classification_report

from typing import Tuple

from face_recog import FaceRecog

from utils import (parse_s3_url, download_and_unzip_dataset, get_mixed_dataset, get_modified_s3_input_paths)
from utils import RS
from utils import (torch, Path, tqdm, DataLoader, cpu_count, device, time, boto3, plt)


torch.manual_seed(RS)


def eval_custom(model, dl: DataLoader) -> Tuple[list, list]:
    y_test, y_pred = [], []
    for imgs, lbls in dl:
        imgs = imgs.to(device)  # torch.Size([16, 3, 256, 256])
        outs = model(imgs)
        _, preds = torch.max(outs, dim=1)
        y_test += lbls.tolist()
        y_pred += preds.tolist()
    return y_test, y_pred


def init_and_launch(n: int, model_name: str, dl: DataLoader, checkpoint: str) -> Tuple[list, list]:
    model = FaceRecog(num_classes=n, model_name=model_name, pretrained=False).to(device)
    model.load_state_dict(torch.load(checkpoint))

    start = time()
    y_test, y_pred = eval_custom(model, dl)
    print(f'\nWasted time on evaluation {model_name} model = {time() - start:.2f} sec')
    return y_test, y_pred


def get_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument("--batch_size", type=int, default=16, help='Batch size for data loader')
    parser.add_argument("--input", nargs='+', type=str, required=True,
                        help='S3 paths to zip files with prepared datasets')
    parser.add_argument("--output", type=str, required=True, help='S3 path to folder for saving logs, errors, scores')

    return parser


def main():
    args = get_parser().parse_args()

    assert len(args.input) == 2, \
        f'########### ERROR: Too much/little inputs {args.input}, only 2 values ###########'

    batch_size = args.batch_size if args.batch_size < 16 else 16

    bucket_output, s3_path_output = parse_s3_url(args.output)
    s3_path_output = s3_path_output.rstrip('/')

    s3_inputs = get_modified_s3_input_paths(args.input)

    checkpoints_folder = Path('models')
    input_datasets_folder = Path('data')
    input_datasets_folder.mkdir(parents=True, exist_ok=True)

    download_and_unzip_dataset(bucket_output, s3_inputs, s3_path_output, input_datasets_folder)

    n_classes, _, standard_dataset = get_mixed_dataset(s3_inputs, input_datasets_folder)

    test_dl = DataLoader(
        standard_dataset, batch_size=batch_size, num_workers=cpu_count(), shuffle=True, pin_memory=True
    )

    all_paths = list(checkpoints_folder.glob('*/*'))
    tqdm_batches = tqdm(all_paths, total=len(all_paths))
    for model in tqdm_batches:
        exp_name = model.parent.name
        model_name = model.stem.split('_checkpoint_')[0]

        tqdm_batches.set_description(f'Evaluating [{model_name}]')
        try:
            y_test, y_pred, = init_and_launch(n_classes, model_name, test_dl, model.as_posix())
        except Exception as error:
            boto3.resource('s3').Object(
                bucket_output,
                f'{s3_path_output}/errors/failed_evaluating_{model_name}.txt'
            ).put(Body=str(error))
            print(f'Report of error in {s3_path_output}/errors/failed_evaluating_{model_name}.txt')
            continue

        file_name = Path(f'{exp_name}_{model_name}.png')

        cm = confusion_matrix(y_test, y_pred, normalize='true')
        cm_df = pd.DataFrame(cm)
        plt.figure(figsize=(50, 50))
        plt.title(f'Confusion Matrix of TOP1 model of {exp_name}')
        sns.heatmap(cm_df, annot=True, cmap='Blues', square=True)
        plt.savefig(file_name.as_posix())

        print(classification_report(y_test, y_pred))
        report = classification_report(y_test, y_pred, output_dict=True)
        with open(f'report_{exp_name}.json', 'w') as f:
            json.dump(report, f)

        for var in (file_name, Path(f'report_{exp_name}.json')):
            boto3.resource('s3').Bucket(bucket_output).upload_file(
                var.as_posix(), f'{s3_path_output}/{exp_name}/{var.name}'
            )


if __name__ == '__main__':
    main()
