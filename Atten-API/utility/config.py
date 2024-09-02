import torch

from utility.parse_args import arg_parse


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

args = arg_parse()
desc_feature_dim = 512
