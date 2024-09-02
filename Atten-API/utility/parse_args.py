import argparse


def arg_parse():
    args = argparse.ArgumentParser()

    # 训练集路径
    args.add_argument('--training_data_path', nargs='?', default='./train data/', type=str)

    # 测试集路径
    args.add_argument('--testing_data_path', nargs='?', default='./test data/', type=str)

    # api文本描述的路径
    args.add_argument('--api_desc_path', nargs='?', default='./origin dataset/API_chatGPT_feature/', type=str)

    # mashup文本描述的路径
    args.add_argument('--mashup_desc_path', nargs='?', default='./origin dataset/mashup_chatGPT_feature/', type=str)

    # 输出结果路径
    args.add_argument('--output_path', nargs='?', default='./output/', type=str)

    # 训练数据集
    args.add_argument('--dataset', nargs='?', default='training_0.json', type=str)

    # 测试数据集
    args.add_argument('--test_dataset', nargs='?', default='testing_2.json', type=str)

    # 模型和训练的参数
    args.add_argument('--n_heads', nargs='?', default=1, type=int)

    args.add_argument('--d_k', nargs='?', default=128, type=int)

    args.add_argument('--d_v', nargs='?', default=128, type=int)

    args.add_argument('--d_q', nargs='?', default=128, type=int)

    args.add_argument('--continue_training', nargs='?', default=1, type=int)

    args.add_argument('--train_batch_size', nargs='?', default=1024, type=int)

    args.add_argument('--test_batch_size', nargs='?', default=64, type=int)

    args.add_argument('--epoch', nargs='?', default=5, type=int)

    args.add_argument('--lr', nargs='?', default=0.01, type=float)

    args.add_argument('--weight_decay', nargs='?', default=0.0001, type=float)

    args.add_argument('--api_range', nargs='?', default=1209, type=int)

    return args.parse_args()
