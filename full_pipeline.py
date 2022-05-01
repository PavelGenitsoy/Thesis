import pandas as pd

from train_and_evaluate import train, evaluate
from utils import *
from face_recog import *


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def fit(
        n_epochs: int, model, train_dl: DataLoader, test_dl: DataLoader, loss_func: nn.CrossEntropyLoss,
        dev: torch.device, optimizer: torch.optim.Adam, train_idx: np.ndarray, valid_idx: np.ndarray,
        model_folder: Path, fold: int
) -> float:
    history = []
    val_loss_ref = float('inf')
    count = 0
    for epoch in range(n_epochs):
        torch.cuda.empty_cache()
        loss, acc = train(epoch, n_epochs, model, train_dl, loss_func, dev, optimizer, train_idx.shape[0])

        torch.cuda.empty_cache()
        val_loss, val_acc = evaluate(model, test_dl, loss_func, dev, valid_idx.shape[0])

        history.append({'loss': loss, 'acc': acc, 'val_loss': val_loss, 'val_acc': val_acc})

        statement = f"[loss]={loss:.4f}; [acc]={acc:.4f}; [val_loss]={val_loss:.4f}; [val_acc]={val_acc:.4f}; "
        print(statement)

        if val_loss < val_loss_ref:
            count = 0
            val_loss_ref = val_loss
        else:
            print(f"[INFO] {count + 1} iteration wasn't significant changes!")
            count += 1

        if count == 3 or (epoch + 1) == n_epochs:
            if not fold:
                check_point_path = Path(f'{model_folder}/checkpoint/{model_folder.name}_checkpoint_{epoch:02d}.pt')
                torch.save(model.state_dict(), check_point_path.as_posix())
                print(f"[INFO] Epoch={epoch + 1}: saving {check_point_path.name}")
            break

    if not fold:
        hist_dict = get_dict_from_list_of_dicts(history)
        pd.DataFrame.from_dict(hist_dict).to_csv(f'{model_folder}/history_last_train.csv', index=False)

    return history[-1]['val_acc']


def init_and_launch(
        ds_classes: int, model_name: str, lr: float, key_path: Path, max_epochs: int, dl: DataLoader,
        val_dl: DataLoader, train_ids: np.ndarray, valid_ids: np.ndarray, fold: int = 0
) -> float:
    model = FaceRecog(num_classes=ds_classes, model_name=model_name).to(device)
    reset_weights(model)
    # model.summary(standard_dataset[0][0].size())

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    start = time()
    result = fit(max_epochs, model, dl, val_dl, criterion, device, optimizer, train_ids, valid_ids, key_path, fold)
    print(f'Wasted time on training {model_name} model = {time() - start:.2f} sec')
    return result


def get_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument("--server", type=int, required=True, help='Number of server')

    parser.add_argument("--models", nargs='+', type=str, required=True, choices=['all', *models_dict.keys()],
                        help='List of models or all, example: --models resnet18 vgg11 | example: --models all')

    parser.add_argument("--batch_size", type=int, default=16, help='Batch size for data loader')

    parser.add_argument("--max_epochs", type=int, default=40, help='Max count of epochs for training')

    parser.add_argument("--mixed", type=bool, default=False,
                        help='if True: generates binomial distribution, gets specific indexes of images from datasets'
                             'by binomial distribution and create concat_dataset from this selected images which is'
                             'used in training'
                             'IMPORTANT ==> works only if you pass to --input s3 paths to two datasets')

    parser.add_argument("--input", nargs='+', type=str, required=True,
                        help='S3 path/s to zip file/s with prepared dataset/s')

    parser.add_argument("--output", type=str, required=True,
                        help='S3 path to folder for saving logs, errors, checkpoints')

    return parser


def main() -> None:
    args = get_parser().parse_args()

    assert len(args.input) in (1, 2), \
        f'########### ERROR: Too much inputs {args.input}, only 1 or 2 value/s ###########'

    if len(args.input) == 1:
        assert not args.mixed, f'########### ERROR: Only 1 input but mixed={args.mixed} its impossible ###########'
    elif len(args.input) == 2:
        assert args.mixed, f'########### ERROR: 2 inputs but mixed={args.mixed} its impossible ###########'

    if 'all' in args.models:
        selected_models = models_dict.keys()
    else:
        selected_models = args.models

    learning_rate = 1e-4
    # mean, std = [0.516, 0.419, 0.373], [0.261, 0.235, 0.222]
    dict_for_csv_cross_valid = {}

    bucket_output, s3_path_output = parse_s3_url(args.output)
    s3_path_output = s3_path_output.rstrip('/')
    s3_path_output = f'{s3_path_output}/server_{args.server:02d}'

    s3_inputs = get_modified_s3_input_paths(args.input)

    train_path = Path('data/train')
    input_datasets_folder = Path('data/input')
    input_datasets_folder.mkdir(parents=True, exist_ok=True)

    download_and_unzip_dataset(bucket_output, s3_inputs, s3_path_output, input_datasets_folder)

    if not args.mixed:
        _, s3_path_input = s3_inputs[0]
        standard_dataset = ImageFolder(
            (input_datasets_folder / s3_path_input.stem).as_posix(), transform=T.ToTensor()
        )

        n_classes = len(standard_dataset.classes)
        targets = np.array(standard_dataset.targets)
    else:
        n_classes, binomial_distribution, standard_dataset = get_mixed_dataset(s3_inputs, input_datasets_folder)

        n_unique_targets, targets = get_targets_for_mixed_dataset(standard_dataset, binomial_distribution)

        assert n_classes == n_unique_targets.shape[0], \
            '########### ERROR: different count of classes after get mixed dataset and targets ###########'

    train_idx, test_idx = split_ds(len(standard_dataset), targets)

    tq_model = tqdm(selected_models, total=len(selected_models))
    for key in tq_model:
        tq_model.set_description(f'Use model [{key}]')

        history_acc = []

        key_path = train_path / Path(key)
        (key_path / Path('checkpoint')).mkdir(parents=True, exist_ok=True)

        for fold, (fold_train_ids, fold_valid_ids) in enumerate(cross_validation(train_idx, N_FOLDS), start=1):
            print('--------------------------------')
            print(f'FOLD {fold}')
            print('--------------------------------')

            dl, val_dl = get_dls(standard_dataset, fold_train_ids, fold_valid_ids, args.batch_size)

            try:
                fold_eval_acc = init_and_launch(
                    n_classes, key, learning_rate, key_path, args.max_epochs, dl, val_dl, fold_train_ids,
                    fold_valid_ids, fold
                )

                history_acc.append(fold_eval_acc)
            except Exception as error:
                boto3.resource('s3').Object(
                    bucket_output,
                    f'{s3_path_output}/errors/fold_{fold}/failed_cross_val_training_{key}.txt'
                ).put(Body=str(error))
                print(f'Report of error in {s3_path_output}/errors/fold_{fold}/failed_cross_val_training_{key}.txt')
                continue

        history_acc = add_mean_std_to_list(history_acc)

        dl, test_dl = get_dls(standard_dataset, train_idx, test_idx, args.batch_size)

        print('--------------------------------')
        print('MAIN TRAIN')
        print('--------------------------------')
        try:
            test_acc = init_and_launch(
                n_classes, key, learning_rate, key_path, args.max_epochs, dl, test_dl, train_idx, test_idx
            )
        except Exception as error:
            boto3.resource('s3').Object(
                bucket_output,
                f'{s3_path_output}/errors/failed_training_{key}.txt'
            ).put(Body=str(error))
            print(f'Report of error in {s3_path_output}/errors/failed_training_{key}.txt')
            continue

        history_acc.append(test_acc)

        dict_for_csv_cross_valid[key] = history_acc

        upload_dire_to_s3(bucket_output, s3_path_output, key_path)

    file_name = train_path / Path('history_cross_validation.csv')

    pd.DataFrame.from_dict(dict_for_csv_cross_valid).to_csv(file_name.as_posix(), index=False)
    boto3.resource('s3').Bucket(bucket_output).upload_file(file_name.as_posix(), f'{s3_path_output}/{file_name.name}')


def test():
    learning_rate = 1e-4

    all_models = models_dict.keys()

    train_path = Path('data/train')
    input_datasets_folder = Path('data/input')
    input_datasets_folder.mkdir(parents=True, exist_ok=True)

    # bucket_input, s3_path_input = parse_s3_url('s3....')
    # s3_path_input = Path(s3_path_input.rstrip('/'))

    # etalon_dataset_s3_key = Path(etalon_datasets[s3_path_input.stem])

    # main_dataset = ImageFolder(
    #     (input_datasets_folder / etalon_dataset_s3_key.stem).as_posix(), transform=T.ToTensor()
    # )

    for dire in input_datasets_folder.glob('*'):
        if dire.is_dir() and dire.name == 'standardized_with_normalized_105_pins':
            # standard_images = Path('data/input/standardized_with_normalized_105_pins')
            # standard_images_list = sorted(list(dire.rglob('*/*')))
            #
            # indexes = [random.randint(0, len(standard_images_list) - 1) for _ in range(4)]
            # show_photos(get_photos([standard_images_list[idx] for idx in indexes]), (2, 2))

            main_dataset = ImageFolder('data/input/105_classes_pins_dataset', transform=T.ToTensor())
            standard_dataset = ImageFolder(dire.as_posix(), transform=T.ToTensor())

            # arr = get_numpy_from_tenzor(standard_dataset[5469][0])
            # image = Image.fromarray((arr * 255).astype(np.uint8)).convert('RGB')
            # image.show()
            # break

            # arr = get_numpy_from_tenzor(denormalize(standard_dataset[indexes[0]][0], *(mean, std)))
            # image = Image.fromarray((arr * 255).astype(np.uint8)).convert('RGB')
            # image.show()
            # break

            classes = standard_dataset.classes

            train_idx, test_idx = split_ds(standard_dataset)
            dl, val_dl = get_dls(standard_dataset, train_idx, test_idx, 32)

            # arr = get_numpy_from_tenzor(val_dl.dataset[2072][0])  # 10291
            # image = Image.fromarray((arr * 255).astype(np.uint8)).convert('RGB')
            # image.show()
            # break

            # tq_model = tqdm(all_models, total=len(all_models))
            # key_path = train_path / Path(key)
            # (key_path / Path('checkpoints')).mkdir(parents=True, exist_ok=True)
            # train
            print(torch.cuda.is_available())
            for key in models_dict.keys():
                torch.cuda.empty_cache()
                model = FaceRecog(num_classes=len(standard_dataset.classes), model_name=key).to(device)
                # layer_to_reset = list(model.children())[-1][-1]

                reset_weights(model)
            # model.apply(reset_weights)
            # size = standard_dataset[0][0].size()
            # model.summary(size)

            # model.load_state_dict(torch.load('data/resnet18_checkpoint_01.pt'))
            #
            # batch_indexes = []
            # y_test, y_pred = [], []
            # for batch_idx, (imgs, lbls, idxs) in tqdm(enumerate(val_dl)):
            #     print(f'Batch idx {batch_idx}, imgs_idx = {idxs}')
            #
            #     imgs = imgs.to(device)  # torch.Size([16, 3, 256, 256])
            #     outs = model(imgs)
            #     _, preds = torch.max(outs, dim=1)
            #     y_test += lbls.tolist()
            #     y_pred += preds.tolist()
            #
            #     print(list(classes[y_test[i + batch_idx * 16]] for i in range(16)))
            #     print(list(classes[y_pred[i + batch_idx * 16]] for i in range(16)))
            #     show_photos(get_normal_photos(idxs, main_dataset), (4, 4))
            #
            #     batch_indexes.append(idxs.numpy().astype(int).tolist())
            #
            #     print(batch_indexes)
            #     print(len(batch_indexes))
            #     if batch_idx == 1:
            #         break
            # pd.DataFrame(np.array(batch_indexes)).to_csv(f'{key_path}/history.csv', index=False)

            # criterion = nn.CrossEntropyLoss()
            #
            # loss, acc = evaluate(model, val_dl, criterion, device, test_idx.shape[0])
            # print(f'loss: {loss}; acc: {acc}')
            #
            # print(classes[y_test[0]], classes[y_test[1]])
            # print(classes[y_pred[0]], classes[y_pred[1]])

            break


if __name__ == '__main__':
    main()
    # test()
