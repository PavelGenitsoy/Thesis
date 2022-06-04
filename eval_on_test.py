import argparse
import pandas as pd

from train_and_evaluate import evaluate
from face_recog import FaceRecog, nn

from utils import (parse_s3_url, download_and_unzip_dataset, get_mixed_dataset, get_modified_s3_input_paths)
from utils import RS
from utils import (torch, Path, tqdm, ConcatDataset, DataLoader, cpu_count, device, time, boto3, np)


torch.manual_seed(RS)
np.random.seed(RS)


def download_checkpoints(exp: str, models_folder: Path):
    bucket, prefix = parse_s3_url(exp)
    prefix = prefix.rstrip('/')

    response = boto3.client('s3').list_objects_v2(Bucket=bucket, Prefix=prefix)

    assert response.get('Contents'), '####### ERROR: Empty response, maybe incorrect path on s3 #######'

    count = 0
    for i in response['Contents']:
        key = i['Key']
        if key.endswith('/'):
            continue
        elif key.endswith('.pt'):
            count += 1
            key = Path(key)

            path_to_save = models_folder / Path(key.parent.parent.name)
            path_to_save.mkdir(parents=True, exist_ok=True)

            boto3.client('s3').download_file(bucket, key.as_posix(), f'{path_to_save}/{key.name}')
            print(f'Downloaded {count}/50 models!')


def init_and_launch(n: int, ds: ConcatDataset, model_name: str, dl: DataLoader, checkpoint: str) -> float:
    model = FaceRecog(num_classes=n, model_name=model_name, pretrained=False).to(device)
    model.load_state_dict(torch.load(checkpoint))

    criterion = nn.CrossEntropyLoss()

    start = time()
    _, result = evaluate(model, dl, criterion, device, len(ds))
    print(f'\nWasted time on evaluation {model_name} model = {time() - start:.2f} sec')
    return result


def get_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument("--batch_size", type=int, default=16, help='Batch size for data loader')
    parser.add_argument("--exp", type=str, required=True,
                        help='S3 path to folder which contain all pretrained models on some dataset')
    parser.add_argument("--input", nargs='+', type=str, required=True,
                        help='S3 paths to zip files with prepared datasets')
    parser.add_argument("--output", type=str, required=True, help='S3 path to folder for saving logs, errors, scores')

    return parser


def main():
    args = get_parser().parse_args()

    assert len(args.input) == 2, \
        f'########### ERROR: Too much/little inputs {args.input}, only 2 values ###########'

    batch_size = args.batch_size if args.batch_size < 16 else 16
    dict_test_acc = {}

    bucket_output, s3_path_output = parse_s3_url(args.output)
    s3_path_output = s3_path_output.rstrip('/')

    s3_inputs = get_modified_s3_input_paths(args.input)

    models_folder = Path('models')
    input_datasets_folder = Path('data')
    input_datasets_folder.mkdir(parents=True, exist_ok=True)

    download_and_unzip_dataset(bucket_output, s3_inputs, s3_path_output, input_datasets_folder)
    download_checkpoints(args.exp, models_folder)

    n_classes, _, standard_dataset = get_mixed_dataset(s3_inputs, input_datasets_folder)

    test_dl = DataLoader(
        standard_dataset, batch_size=batch_size, num_workers=cpu_count(), shuffle=True, pin_memory=True
    )

    all_paths = list(models_folder.glob('*/*'))
    tqdm_batches = tqdm(all_paths, total=len(all_paths))
    for model in tqdm_batches:
        model_name = model.parent.name

        tqdm_batches.set_description(f'Evaluating [{model_name}]')
        try:
            test_acc = init_and_launch(n_classes, standard_dataset, model_name, test_dl, model.as_posix())
        except Exception as error:
            boto3.resource('s3').Object(
                bucket_output,
                f'{s3_path_output}/errors/failed_evaluating_{model_name}.txt'
            ).put(Body=str(error))
            print(f'Report of error in {s3_path_output}/errors/failed_evaluating_{model_name}.txt')
            continue

        dict_test_acc[model_name] = test_acc

    file_name = Path('history_scores_on_test.csv')

    df = pd.DataFrame.from_dict([dict_test_acc])
    df.sort_values(by=df.index[-1], axis=1, ascending=False, inplace=True)
    df.to_csv(file_name.as_posix(), index=False)
    boto3.resource('s3').Bucket(bucket_output).upload_file(file_name.as_posix(), f'{s3_path_output}/{file_name.name}')


if __name__ == '__main__':
    main()
