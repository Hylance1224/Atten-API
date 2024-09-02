import os
import re
import json
import torch
import model_atten_API
import utility.config

import numpy as np
import utility.metrics as metrics

from typing import Dict, List
from utility.transcoding import encode_api_context

args = utility.config.args
ks: List = [1, 3, 5, 10]                          # 推荐指标的top_n


def encode_test_data(data: Dict) -> Dict:
    api_list = data['api_list']

    batch_mashup_description_feature = []
    batch_api_description_feature = []
    batch_context_len = []

    encoded_api_context = encode_api_context(api_list=api_list)     # encoded_api_context: [used_api_num, 512]
    mashup_description_feature = np.loadtxt('%s%d.txt' % (args.mashup_desc_path, data['mashup_id']))
    for i in range(args.api_range):
        target_api = i + 1
        if target_api not in api_list:
            api_description_feature = np.loadtxt(args.api_desc_path + str(target_api) + '.txt')

            batch_api_description_feature.append(api_description_feature)
            batch_mashup_description_feature.append(mashup_description_feature)
            batch_context_len.append(len(encoded_api_context))

    input_data = {
        'encoded_api_context': torch.tensor(encoded_api_context, dtype=torch.float32).repeat(len(batch_mashup_description_feature), 1, 1),  # encoded_api_context: [batch_size, used_api_num,512]
        'mashup_description_feature': torch.tensor(batch_mashup_description_feature, dtype=torch.float32),      # mashup_description_feature: [batch_size, 512]
        'candidate_api_description_feature': torch.tensor(batch_api_description_feature, dtype=torch.float32),   # api_description_feature: [batch_size, 512]
        'context_len': torch.tensor(batch_context_len)
    }

    return input_data


def get_top_n_api(probability_list, top_n) -> List:
    p_list = probability_list
    p_list = sorted(p_list, reverse=True)
    top_n_api = []
    for i in range(top_n):
        top_n_api.append(p_list[i][1])
    return top_n_api


def get_performance(user_pos_test, r, auc) -> Dict:
    precision, recall, ndcg, \
    map, fone, mrr = [], [], [], [], [], []

    for k in ks:
        precision.append(metrics.precision_at_k(r, k))
        recall.append(metrics.recall_at_k(r, k, len(user_pos_test)))
        ndcg.append(metrics.ndcg_at_k(r, k, 1))
        map.append(metrics.average_precision(r, k))
        mrr.append(metrics.mrr_at_k(r, k))
        fone.append(metrics.F1(metrics.precision_at_k(r, k), metrics.recall_at_k(r, k, len(user_pos_test))))

    return {
        'recall': np.array(recall), 'precision': np.array(precision), 'ndcg': np.array(ndcg),
        'map': np.array(map), 'mrr': np.array(mrr), 'fone': np.array(fone)
    }


def test_one(pos_test, user_rating) -> Dict:
    r = []
    for i in user_rating:
        if i in pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = 0.
    return get_performance(pos_test, r, auc)


def test_model(model_path: str) -> None:

    recommend_file = args.output_path + output_file

    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)
    # 推荐结果指针
    write_recommend_fp = open(file=recommend_file, mode='w')
    # 读取测试文件指针
    test_fp = open(file=args.testing_data_path + test_data, mode='r')

    model = model_atten.WideAndDeep()
    model.load_state_dict(torch.load(model_path))

    model = model.to(utility.config.device)
    result_list: List = []

    test_num: int = 0
    model.eval()
    with torch.no_grad():
        for lines in test_fp.readlines():
            test_obj = json.loads(lines.strip('\n'))
            test_num += 1
            inputs = encode_test_data(test_obj)

            outputs = model(inputs)
            outputs = outputs.view(-1).tolist()
            probability_list = []

            removed_api = test_obj['removed_api_list']

            api_list = test_obj['api_list']

            num = 0
            for i in range(args.api_range):
                target_api = i + 1
                if target_api not in api_list:
                    probability_list.append((outputs[num], target_api))
                    num += 1

            top_n_api = get_top_n_api(probability_list, 20)
            write_data = {
                'mashup_id': test_obj['mashup_id'],
                'recommend_api': top_n_api,
                'removed_apis': test_obj['removed_api_list']
            }
            write_content = json.dumps(write_data) + '\n'
            write_recommend_fp.write(write_content)

            set_true = set(removed_api) & set(top_n_api[:10])
            list_true = list(set_true)

            result_list.append(len(list_true) / 10.0)

            result = sum(result_list) / len(result_list)
            print(test_num)
            print(result)
            print('--------------------')

    test_fp.close()
    write_recommend_fp.close()


if __name__ == '__main__':
    path: str = 'model_atten'
    fold: str = '4'
    rm = '2'
    output_file = f'test_Atten_{fold}_{rm}.json'
    test_data = f'testing_{fold}_{rm}.json'
    test_model(f'./{path}/model_{fold}.pth')
