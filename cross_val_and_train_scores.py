import pandas as pd

from utils import *


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument("--input", nargs='+', type=str, required=True,
                        help='S3 path/s to folder, which contains results about training')

    return parser


def main() -> None:
    args = get_parser().parse_args()

    local_folder = Path('data/cross_val_and_last_train_scores')

    s3_inputs = get_modified_s3_input_paths(args.input)

    for s3_tuple in s3_inputs:
        get_cross_val_from_s3(s3_tuple[0], s3_tuple[1], local_folder)

    for subfolder in local_folder.glob('*'):
        if subfolder.is_dir():
            df_main = pd.DataFrame([], dtype=np.float64)
            for file in subfolder.glob('*.csv'):
                tmp = pd.read_csv(file, dtype=np.float64)
                df_main = pd.concat([df_main, tmp], axis=1)

            df_main.sort_values(by=df_main.index[-1], axis=1, ascending=False, inplace=True)
            df_main.to_csv(f'{subfolder}/history_cross_validation_sorted_by_train_scores.csv', index=False)
            print(f'{subfolder.name}: combining dataframes and sorting by scores on the train are done!')


if __name__ == '__main__':
    main()
