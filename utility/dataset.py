import json
import torch
import utility.transcoding
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from utility.parse_args import arg_parse
args = arg_parse()



def collate_fn(batch_data_list):
    encoded_api_context_inputs = [torch.from_numpy(batch_data[0]['encoded_api_context']) for batch_data in batch_data_list]
    max_len = len(max(encoded_api_context_inputs, key=lambda x: len(x)))
    encoded_api_context_inputs = list(map(lambda x: torch.cat((x, torch.zeros(size=(max_len-len(x), args.desc_feature_dim)).double()), dim=0) if len(x) < max_len else x,
                                     encoded_api_context_inputs))

    encoded_api_context = torch.stack([api_context for api_context in encoded_api_context_inputs], dim=0).double()

    mashup_description_feature = torch.stack([torch.from_numpy(batch_data[0]['mashup_description_feature']) for batch_data in batch_data_list], dim=0).double()
    candidate_api_description_feature = torch.stack([torch.from_numpy(batch_data[0]['candidate_api_description_feature']) for batch_data in batch_data_list], dim=0).double()

    mashup_tags = torch.stack(
        [torch.from_numpy(batch_data[0]['mashup_tags']) for batch_data in batch_data_list],
        dim=0)
    target_api_tags = torch.stack(
        [torch.from_numpy(batch_data[0]['target_api_tags']) for batch_data in batch_data_list],
        dim=0).double()
    cross_product_tag = torch.stack(
        [torch.from_numpy(batch_data[0]['cross_product_tag']) for batch_data in batch_data_list],
        dim=0).double()
    labels = torch.stack([torch.tensor(batch_data[1]) for batch_data in batch_data_list], dim=0).double()
    context_lens_tensor = torch.stack([torch.tensor(batch_data[0]['context_len']) for batch_data in batch_data_list], dim=0)

    inputs = {
        'encoded_api_context': encoded_api_context,
        'mashup_description_feature': mashup_description_feature,
        'candidate_api_description_feature': candidate_api_description_feature,
        'context_len': context_lens_tensor,
        'mashup_tags': mashup_tags,
        'target_api_tags': target_api_tags,
        'cross_product_tag': cross_product_tag
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
        # print(data)
        label = data['label']
        data = utility.transcoding.encode_data(data)

        return data, label


def get_dataloader(train: bool = True) -> DataLoader:
    dataset = APIDataSet()
    batch_size = args.train_batch_size if train else args.test_batch_size
    loader = DataLoader(dataset=dataset, shuffle=True, batch_size=batch_size, num_workers=2, collate_fn=collate_fn)

    return loader
