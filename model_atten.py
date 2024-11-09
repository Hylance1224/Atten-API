import torch
import torch.nn as nn
from utility.parse_args import arg_parse
args = arg_parse()


class AttentionModule(nn.Module):
    def __init__(self):
        super(AttentionModule, self).__init__()
        self.W_K = nn.Linear(args.desc_feature_dim, args.n_heads * args.d_k, bias=False)
        self.W_Q = nn.Linear(args.desc_feature_dim, args.n_heads * args.d_q, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_k, input_q, context_len):
        batch_size = input_k.shape[0]
        K = self.W_K(input_k).reshape(batch_size, -1, args.n_heads, args.d_k).transpose(1, 2)
        Q = self.W_Q(input_q).reshape(batch_size, -1, args.d_q)
        Q = torch.unsqueeze(Q, dim=2)

        QK_t = torch.matmul(Q, K.transpose(-1, -2))
        QK_t_new = QK_t.squeeze(dim=(1, 2))

        result_list = []
        for num in range(input_k.shape[0]):
            length = int(context_len[num].item())
            if length == 0:
                size = torch.Size([512])
                sum_result = torch.zeros(size)
                result_list.append(sum_result)
            else:
                QK_t_new_num_part = QK_t_new[num][:length]
                attention_score_part = self.softmax(QK_t_new_num_part)
                input_k_num = input_k[num][:length]
                sum_result = torch.sum(attention_score_part.unsqueeze(1) * input_k_num, dim=0)
                result_list.append(sum_result)

        result = torch.stack(result_list)
        result = torch.unsqueeze(result, dim=1)
        context = result
        context = context.reshape(batch_size, -1)

        return context


class WideAndDeep(nn.Module):
    def __init__(self):
        super(WideAndDeep, self).__init__()

        self.wide = nn.Sequential(
            nn.Linear(args.tag_range + args.tag_range + args.tag_range * args.tag_range, 1),
            nn.Sigmoid()
        )

        self.attn_layer = AttentionModule()

        self.deep = nn.Sequential(
            nn.Linear(512 + 512 + 512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.wide_weight = nn.Parameter(torch.ones(1))
        self.deep_weight = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))
        self.sigmoid_act_fun = nn.Sigmoid()

    def forward(self, input):
        wide_inputs = [input['mashup_tags'], input['target_api_tags'],
                       input['cross_product_tag']]
        wide_input = torch.cat(wide_inputs, dim=1).float()
        wide_input = wide_input.to(args.device)
        wide_output = self.wide(wide_input)

        input_k, input_q = input['encoded_api_context'], input['candidate_api_description_feature']
        input_k = input_k.to(args.device).float()
        input_q = input_q.to(args.device).float()

        api_context = self.attn_layer(input_k, input_q, input['context_len'])

        mashup_description_feature = input['mashup_description_feature']
        mashup_description_feature = mashup_description_feature.to(args.device).float()

        deep_inputs = [mashup_description_feature, api_context, input_q]
        deep_input = torch.cat(deep_inputs, dim=1)
        deep_output = self.deep(deep_input)

        out = self.wide_weight * wide_output + self.deep_weight * deep_output + self.bias

        output = self.sigmoid_act_fun(out)

        return output
