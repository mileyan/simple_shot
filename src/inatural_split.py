import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, help='path to the INat dataset')
parser.add_argument('--split', type=str, default='split/inatural', help='path to the INat dataset')


def walk_path(data, tag):
    results = []
    path = '{}/{}/'.format(data, tag)
    sub_folders = os.listdir(path)
    for sub in sorted(sub_folders):
        sub_path = '{}/{}'.format(path, sub)
        files = [('{}/{}/{}'.format(tag, sub, x), sub) for x in sorted(os.listdir(sub_path))]
        results += files
    return results


if __name__ == '__main__':
    args = parser.parse_args()
    if not os.path.isdir(args.split):
        os.makedirs(args.split)
    with open('{}/train.csv'.format(args.split), 'w') as f:
        samples = walk_path(args.data, 'train')
        for i, j in samples:
            f.write('{},{}\n'.format(i, j))
    os.system('cp {0}/train.csv {0}/val.csv'.format(args.split))
    with open('{}/repr.csv'.format(args.split), 'w') as f:
        samples = walk_path(args.data, 'repr')
        for i, j in samples:
            f.write('{},{}\n'.format(i, j))
    with open('{}/query.csv'.format(args.split), 'w') as f:
        samples = walk_path(args.data, 'query')
        for i, j in samples:
            f.write('{},{}\n'.format(i, j))
