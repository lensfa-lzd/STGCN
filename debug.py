import argparse


def get_parameters():
    parser = argparse.ArgumentParser(description='STGCN')
    parser.add_argument('--enable_nni', type=bool, default=False, help='enable nni experiment')
    parser.add_argument('--nni', type=int, default=123, help='enable nni experiment')

    args = parser.parse_args()
    print(args)

    RCV_CONFIG = {'batch_size': 16, 'optimizer': 'Adam'}
    parser.set_defaults(**RCV_CONFIG)
    args = parser.parse_args()
    print(args)


    return args


if __name__ == '__main__':
    # Logging
    args = get_parameters()

    RCV_CONFIG = {'batch_size': 16, 'optimizer': 'Adam'}


