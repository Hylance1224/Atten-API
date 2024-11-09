import os
import json
import torch
import model_atten
import numpy as np
from typing import Dict, List
from utility.transcoding import encode_api_context, cross_product_transformation_torch, encode_tag_list_torch
from utility.parse_args import arg_parse
args = arg_parse()

import pickle
ks: List = [1, 3, 5, 10]


def read_json_file(path):
    f = open(path, 'r')
    lines = f.readlines()
    datas = []
    for line in lines:
        data = json.loads(line.strip())
        datas.append(data)
    f.close()
    return datas


with open('origin dataset/tag_id.pickle', 'rb') as f:
    tag_id_mapping = pickle.load(f)
global_mashups = read_json_file('origin dataset/mashup_details.json')
global_apis = read_json_file('origin dataset/api_details.json')

def tag_2_id(tag_list, tag_id_mapping):
    tag_list = tag_list.split(',')
    tag_id_list = []
    for tag in tag_list:
        tag_id = [item['tag_id'] for item in tag_id_mapping if item['tag'] == tag][0]
        tag_id_list.append(tag_id)
    return tag_id_list


def obtain_tags_for_mashup(mashup_id):
    mashup_tags = [item['mashup_tag'] for item in global_mashups if item['mashup_id'] == mashup_id][0]
    mashup_tags = tag_2_id(mashup_tags, tag_id_mapping)
    return mashup_tags


def obtain_tags_for_api(api_id):
    tags = [item['tag'] for item in global_apis if item['api_id'] == api_id][0]
    tags = tag_2_id(tags, tag_id_mapping)
    return tags

def encode_test_data(data: Dict) -> Dict:
    api_list = data['api_list']

    batch_mashup_description_feature = []
    batch_api_description_feature = []
    batch_context_len = []
    batch_mashup_tags = []
    batch_api_tags = []
    batch_cross_product_tags = []

    encoded_api_context = encode_api_context(api_list=api_list)     # encoded_api_context: [used_api_num, 512]
    mashup_description_feature = np.loadtxt('%s%d.txt' % (args.mashup_desc_path, data['mashup_id']))
    mashup_tags = obtain_tags_for_mashup(data['mashup_id'])
    mashup_tags = encode_tag_list_torch(mashup_tags)

    for i in range(args.api_range):
        target_api = i + 1
        if target_api not in api_list:
            api_description_feature = np.loadtxt(args.api_desc_path + str(target_api) + '.txt')
            api_tags = obtain_tags_for_api(target_api)
            api_tags = encode_tag_list_torch(api_tags)
            cross_product_tags = cross_product_transformation_torch(mashup_tags, api_tags)

            batch_api_description_feature.append(api_description_feature)
            batch_mashup_description_feature.append(mashup_description_feature)
            batch_context_len.append(len(encoded_api_context))
            batch_mashup_tags.append(mashup_tags)
            batch_api_tags.append(api_tags)
            batch_cross_product_tags.append(cross_product_tags)

    input_data = {
        'encoded_api_context': torch.tensor(encoded_api_context, dtype=torch.float32).repeat(len(batch_mashup_description_feature), 1, 1),  # encoded_api_context: [batch_size, used_api_num,512]
        'mashup_description_feature': torch.tensor(batch_mashup_description_feature, dtype=torch.float32),      # mashup_description_feature: [batch_size, 512]
        'candidate_api_description_feature': torch.tensor(batch_api_description_feature, dtype=torch.float32),   # api_description_feature: [batch_size, 512]
        'context_len': torch.tensor(batch_context_len),
        'mashup_tags': torch.stack(batch_mashup_tags),
        'target_api_tags': torch.stack(batch_api_tags),
        'cross_product_tag': torch.stack(batch_cross_product_tags)
    }
    return input_data


def get_top_n_api(probability_list, top_n) -> List:
    p_list = probability_list
    p_list = sorted(p_list, reverse=True)
    top_n_api = []
    for i in range(top_n):
        top_n_api.append(p_list[i][1])
    return top_n_api


def test_model(model_path: str) -> None:
    recommend_file = args.output_path + output_file

    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)
    write_recommend_fp = open(file=recommend_file, mode='w')
    test_fp = open(file=args.testing_data_path + test_data, mode='r')

    model = model_atten.WideAndDeep()
    model.load_state_dict(torch.load(model_path))

    model = model.to(args.device)

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
    test_fp.close()
    write_recommend_fp.close()


if __name__ == '__main__':
    path: str = 'model_atten'
    fold: str = '0'
    rm = '1'
    output_file = f'test_Atten_{fold}_{rm}.json'
    test_data = f'testing_{fold}_{rm}.json'
    test_model(f'./{path}/model_{fold}.pth')
