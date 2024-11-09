import random
import json
import os
from utility.parse_args import arg_parse
import copy

args = arg_parse()

fold_num = 5
removed_nums = [1, 2]


def read_json_relation_file(path):
    f = open(path, 'r')
    lines = f.readlines()
    datas = []
    for line in lines:
        data = json.loads(line.strip())
        datas.append(data)
    f.close()
    return datas


def generate_training_data(AppID_TplList, file):
    f = open(file, 'w', encoding='utf-8')
    for data in AppID_TplList:
        app_id = data['mashup_id']
        tpl_list = data['api_list']
        for i in range(0, len(tpl_list)):
            # generate true sample
            new_tpl_list = tpl_list.copy()
            new_tpl_list.pop(i)
            temp = {'mashup_id': app_id, 'api_list': new_tpl_list, 'target_api': tpl_list[i], 'label': 1}

            js = json.dumps(temp, ensure_ascii=False)
            f.write(js)
            f.write('\n')

            # generate false sample
            false_tpl_list = [x for x in range(1, args.api_range + 1) if x not in tpl_list]
            false_tpls = random.choices(false_tpl_list, k=10)
            for i in false_tpls:
                false_tpl = i
                temp = {'mashup_id': app_id, 'api_list': new_tpl_list, 'target_api': false_tpl,
                         'label': 0}
                js = json.dumps(temp, ensure_ascii=False)
                f.write(js)
                f.write('\n')
    f.close()


def generate_testing_data(AppID_ApiList, remove_num, file):
    f = open(file, 'w', encoding='utf-8')
    new_list = copy.deepcopy(AppID_ApiList)
    for data in new_list:
        temp_data = data
        app_id = temp_data['mashup_id']
        api_list = temp_data['api_list']
        if len(api_list) >= remove_num:
            removed_api_list = random.sample(api_list, remove_num)
            for r in removed_api_list:
                api_list.remove(r)
            temp = {'mashup_id': app_id, 'api_list': api_list, 'removed_api_list': removed_api_list}
            js = json.dumps(temp, ensure_ascii=False)
            f.write(js)
            f.write('\n')
    f.close()


def main_training(json_relation_file='./origin dataset/relation.json', n_fold=fold_num):
    AppID_TplList = read_json_relation_file(json_relation_file)
    length = len(AppID_TplList)
    fold_num = int(length/n_fold)
    for i in range(n_fold):
        datas = AppID_TplList[0:fold_num*i]
        datas.extend(AppID_TplList[fold_num*(i+1):])
        generate_training_data(datas, args.training_data_path + '/training_' + str(i) + '.json')


def main_testing(json_relation_file='./origin dataset/relation.json', n_fold=fold_num):
    AppID_TplList = read_json_relation_file(json_relation_file)
    length = len(AppID_TplList)
    fold_num = int(length/n_fold)
    for i in range(n_fold):
        for j in removed_nums:
            temp = AppID_TplList[fold_num * i: fold_num * (i + 1)]
            generate_testing_data(temp, remove_num=j,
                                  file=args.testing_data_path + '/testing_' + str(i) + '_' + str(j) + '.json')


if __name__ == '__main__':
    if not os.path.exists(args.training_data_path):
        os.mkdir(args.training_data_path)
    if not os.path.exists(args.testing_data_path):
        os.mkdir(args.testing_data_path)
    main_training()
    main_testing()


