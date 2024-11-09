import numpy as np
import torch
import h5py
from typing import Dict, Union, List
from utility.parse_args import arg_parse
args = arg_parse()


def encode_api_list(api_list, api_range=args.api_range) -> np.ndarray:
    encoded_api = np.zeros(api_range)
    for api in api_list:
        encoded_api[api - 1] = 1
    return encoded_api


def encode_tag_list(tag_list, tag_range=args.tag_range) -> np.ndarray:
    encoded_tag = np.zeros(tag_range)
    for tag in tag_list:
        encoded_tag[tag - 1] = 1
    return encoded_tag


def encode_tag_list_torch(tag_list, tag_range: int = args.tag_range):
    encoded_tag = torch.zeros(tag_range, dtype=torch.float32).to(args.device)
    for tag in tag_list:
        encoded_tag[tag - 1] = 1
    return encoded_tag


def encode_api(api, api_range: int = args.api_range) -> np.ndarray:
    encoded_api = np.zeros(api_range)
    encoded_api[api - 1] = 1
    return encoded_api


def encode_api_context(api_list) -> np.ndarray:
    if len(api_list) == 0:
        return np.zeros(shape=(1, args.desc_feature_dim))
    return np.array([get_vector_by_id(api, args.api_desc_path) for api in api_list])


def cross_product_transformation(encoded_api_list, encoded_api) -> np.ndarray:
    cross_product_feature = np.array([x1*x2 for x1 in encoded_api_list for x2 in encoded_api])
    return cross_product_feature


def cross_product_transformation_torch(encoded_api_list: torch.Tensor, encoded_api: torch.Tensor) -> torch.Tensor:
    encoded_api_list = encoded_api_list.to(args.device).unsqueeze(1)
    encoded_api = encoded_api.to(args.device).unsqueeze(0)
    cross_product_feature = torch.matmul(encoded_api_list, encoded_api).flatten()
    return cross_product_feature


def get_vector_by_id(id, file):
    with h5py.File(file, 'r') as f:
        if str(id) in f:
            return f[str(id)][:]
        else:
            return None


def encode_data(data: Dict) -> Dict[str, Union[List, np.ndarray, int]]:
    encoded_api_context = encode_api_context(api_list=data['api_list'])
    encoded_mashup_tag = encode_tag_list(data['mashup_tags'])
    encoded_api_tag = encode_tag_list(data['target_api_tags'])
    cross_product_tag = cross_product_transformation(encoded_mashup_tag, encoded_api_tag)
    context_len = len(encoded_api_context)

    mashup_description_feature = get_vector_by_id(data['mashup_id'], args.mashup_desc_path)
    candidate_api_description_feature = get_vector_by_id(data['target_api'], args.api_desc_path)
    data = {
        'encoded_api_context': encoded_api_context,
        'candidate_api_description_feature': candidate_api_description_feature,
        'mashup_description_feature': mashup_description_feature,
        'context_len': np.array([context_len]),
        'mashup_tags': encoded_mashup_tag,
        'target_api_tags': encoded_api_tag,
        'cross_product_tag': cross_product_tag
    }
    return data

