import json

import torch

import utility.transcoding

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

config = utility.transcoding.config
args = utility.transcoding.args


def collate_fn(batch_data_list):        # batch_data_list: [batch_size, tuple(dict(str, np.ndarray), label)]
    encoded_api_context_inputs = [torch.from_numpy(batch_data[0]['encoded_api_context']) for batch_data in batch_data_list]   # encoded_api_context: [num_used_api, 512]    num_used_api 为本batch中api_list的最多的个数
                                                                                                            # encoded_api_context_inputs: [batch_size, num_used_api, 512]
    # 在这里要处理这个encoded_api_context_inputs, 要将没有used_api的填充tensor 0, 填充长度为512(sentence bert的向量化长度)
    max_len = len(max(encoded_api_context_inputs, key=lambda x: len(x)))         # 本batch中最长的使用api list长度
    encoded_api_context_inputs = list(map(lambda x: torch.cat((x, torch.zeros(size=(max_len-len(x), config.desc_feature_dim)).double()), dim=0) if len(x) < max_len else x,
                                     encoded_api_context_inputs))                # encoded_api_context_inputs: [batch_size, num_used_api, 512]，但是是列表，需要转成tensor

    encoded_api_context = torch.stack([api_context for api_context in encoded_api_context_inputs], dim=0).double()   # encoded_api_context: [batch_size, num_used_api, 512] 维度不变，是tensor
    mashup_description_feature = torch.stack([torch.from_numpy(batch_data[0]['mashup_description_feature']) for batch_data in batch_data_list], dim=0).double()
    candidate_api_description_feature = torch.stack([torch.from_numpy(batch_data[0]['candidate_api_description_feature']) for batch_data in batch_data_list], dim=0).double()

    labels = torch.stack([torch.tensor(batch_data[1]) for batch_data in batch_data_list], dim=0).double()     # labels: [batch_size]
    context_lens_tensor = torch.stack([torch.tensor(batch_data[0]['context_len']) for batch_data in batch_data_list], dim=0)

    inputs = {
        'encoded_api_context': encoded_api_context,
        'mashup_description_feature': mashup_description_feature,
        'candidate_api_description_feature': candidate_api_description_feature,
        'context_len': context_lens_tensor

    }
    return inputs, labels


class APIDataSet(Dataset):
    def __init__(self):
        super(APIDataSet, self).__init__()
        file_path: str = args.training_data_path + args.dataset
        number: int = 0
        with open(file=file_path, mode='r') as fp:
            for _ in tqdm(fp, desc='load dataset', leave=False):
                number += 1
        fp = open(file=file_path, mode='r')
        lines = fp.readlines()
        self.size: int = number
        self.file = lines

    def __len__(self):
        return self.size

    def __getitem__(self, item_idx):
        line = self.file[item_idx]

        data = json.loads(line.strip('\n'))
        label = data['label']
        data = utility.transcoding.encode_data(data)

        return data, label


def get_dataloader(train: bool = True) -> DataLoader:
    dataset = APIDataSet()
    batch_size = args.train_batch_size if train else args.test_batch_size
    loader = DataLoader(dataset=dataset, shuffle=True, batch_size=batch_size, num_workers=0, collate_fn=collate_fn)

    return loader
