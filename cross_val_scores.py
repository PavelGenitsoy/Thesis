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


if __name__ == '__main__':
    main()
