import json
import pickle


def read_json_file(path):
    f = open(path, 'r')
    lines = f.readlines()
    datas = []
    for line in lines:
        data = json.loads(line.strip())
        datas.append(data)
    f.close()
    return datas


def get_tag_id():
    mashup_datas = read_json_file('origin dataset/mashup_details.json')
    api_datas = read_json_file('origin dataset/api_details.json')

    all_tags = []
    tag_id = 0

    for mashup in mashup_datas:
        mashup_tags = mashup['mashup_tag']
        mashup_tags = mashup_tags.split(',')
        for tag in mashup_tags:
            if tag not in all_tags:
                all_tags.append(tag)

    for api in api_datas:
        # print(api)
        api_tags = api['tag']
        try:
            api_tags = api_tags.split(',')
            for tag in api_tags:
                if tag not in all_tags:
                    all_tags.append(tag)
        except:
            print(api['api_id'])

    tag_list = []
    num = 1
    for tag in all_tags:
        tag_list.append({'tag_id': num, 'tag': tag})
        num = num + 1

    print(tag_list)

    # 保存变量到文件
    with open('origin dataset/tag_id.pickle', 'wb') as f:
        pickle.dump(tag_list, f)


def tag_2_id(tag_list, tag_id_mapping):
    tag_list = tag_list.split(',')
    tag_id_list = []
    for tag in tag_list:
        tag_id = [item['tag_id'] for item in tag_id_mapping if item['tag'] == tag][0]
        tag_id_list.append(tag_id)
    return tag_id_list


def rename_train_file():
    with open('origin dataset/tag_id.pickle', 'rb') as f:
        tag_id_mapping = pickle.load(f)
    mashups = read_json_file('origin dataset/mashup_details.json')
    apis = read_json_file('origin dataset/api_details.json')
    original_train_datas = read_json_file('train data/training_3.json')
    new_train_f = open('train data/training_3_new.json', mode='w')
    for original_train_data in original_train_datas:
        print(original_train_data)
        mashup_id = original_train_data['mashup_id']
        target_api_id = original_train_data['target_api']
        mashup_tags = [item['mashup_tag'] for item in mashups if item['mashup_id'] == mashup_id][0]
        mashup_tags = tag_2_id(mashup_tags, tag_id_mapping)
        target_api_tags = [item['tag'] for item in apis if item['api_id'] == target_api_id][0]
        target_api_tags = tag_2_id(target_api_tags, tag_id_mapping)
        original_train_data['mashup_tags'] = mashup_tags
        original_train_data['target_api_tags'] = target_api_tags

        write_content = json.dumps(original_train_data) + '\n'
        new_train_f.write(write_content)


if __name__ == '__main__':
    with open('origin dataset/tag_id.pickle', 'rb') as f:
        tag_id_mapping = pickle.load(f)
    print(len(tag_id_mapping))