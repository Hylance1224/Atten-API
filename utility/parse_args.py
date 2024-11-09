import argparse
import torch

def arg_parse():
    args = argparse.ArgumentParser()

    args.add_argument('--training_data_path', nargs='?', default='./train data/', type=str)

    args.add_argument('--testing_data_path', nargs='?', default='./test data/', type=str)

    args.add_argument('--api_desc_path', nargs='?', default='./origin dataset/API_vectors.h5', type=str)

    args.add_argument('--mashup_desc_path', nargs='?', default='./origin dataset/Mashup_vectors.h5', type=str)

    args.add_argument('--output_path', nargs='?', default='./output/', type=str)

    args.add_argument('--dataset', nargs='?', default='training_0.json', type=str)

    args.add_argument('--n_heads', nargs='?', default=1, type=int)

    args.add_argument('--d_k', nargs='?', default=128, type=int)

    args.add_argument('--d_v', nargs='?', default=128, type=int)

    args.add_argument('--d_q', nargs='?', default=128, type=int)

    args.add_argument('--continue_training', nargs='?', default=0, type=int)

    args.add_argument('--train_batch_size', nargs='?', default=1024, type=int)

    args.add_argument('--epoch', nargs='?', default=15, type=int)

    args.add_argument('--lr', nargs='?', default=0.01, type=float)

    args.add_argument('--weight_decay', nargs='?', default=0.0001, type=float)

    args.add_argument('--api_range', nargs='?', default=1209, type=int)

    args.add_argument('--desc_feature_dim', nargs='?', default=512, type=int)

    args.add_argument('--device', nargs='?', default='cuda' if torch.cuda.is_available() else 'cpu', type=str)

    return args.parse_args()
