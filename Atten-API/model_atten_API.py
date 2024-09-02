import torch
import numpy as np
import torch.nn as nn
import utility.config as config


class AttentionModule(nn.Module):
    def __init__(self):
        super(AttentionModule, self).__init__()
        self.W_K = nn.Sequential(
            nn.Linear(config.desc_feature_dim, config.args.n_heads * config.args.d_k),     # W_K: [512, head * d_k]
            nn.Sigmoid()
        )
        self.W_Q = nn.Sequential(
            nn.Linear(config.desc_feature_dim, config.args.n_heads * config.args.d_k),  # W_K: [512, head * d_k]
            nn.Sigmoid()
        )

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_k, input_q, context_len):     # input_k.shape == input_v.shape: [batch_size, max_used_api_num, 512]   input_q: [batch_size, 512]
        batch_size = input_k.shape[0]
        K = self.W_K(input_k).reshape(batch_size, -1, config.args.n_heads, config.args.d_k).transpose(1, 2)    # K: [batch_size, head, max_used_api_num, d_k]
        Q = self.W_Q(input_q).reshape(batch_size, -1, config.args.d_q)                                         # Q: [batch_size, head, d_q]
        # Q在做乘法之前要先unsqueeze一下, 因为torch要求除了最后的两个维度其他的维度要相同, 确保Q能和K进行相乘
        Q = torch.unsqueeze(Q, dim=2)                                     # Q的维度: [batch_size, head, 1, d_q]

        # QK-T
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
                # 对QK-t的前length进行softmax
                attention_score_part = self.softmax(QK_t_new_num_part) # [context_len, 512]
                input_k_num = input_k[num][:length]  # [context_len, 1]

                # 这一行代码实现attention_score依次和API上文相乘
                print(attention_score_part)
                sum_result = torch.sum(attention_score_part.unsqueeze(1) * input_k_num, dim=0)
                result_list.append(sum_result)

        result = torch.stack(result_list)  # torch.Size([256, 763])
        result = torch.unsqueeze(result, dim=1)
        context = result  # context: [batch_size, head, d_v]
        context = context.reshape(batch_size, -1)

        return context


class WideAndDeep(nn.Module):
    def __init__(self):
        super(WideAndDeep, self).__init__()

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

        self.sigmoid_act_fun = nn.Sigmoid()

    def forward(self, input):

        input_k, input_q = input['encoded_api_context'], input['candidate_api_description_feature']
        input_k = input_k.to(config.device).float()                     # input_k: [batch_size, max_used_api_num, 512]                     # input_v: [batch_size, max_used_api_num, 512]
        input_q = input_q.to(config.device).float()                     # input_q: [batch_size, 512]

        # 注意力部分
        api_context = self.attn_layer(input_k, input_q, input['context_len'])        # api_context: [batch_size, head * d_v]

        # deep 部分
        mashup_description_feature = input['mashup_description_feature']
        mashup_description_feature = mashup_description_feature.to(config.device).float()

        deep_inputs = [mashup_description_feature, api_context, input_q]
        deep_input = torch.cat(deep_inputs, dim=1)                      # deep_input: [batch_size, head*dv + 512]
        deep_output = self.deep(deep_input)                             # deep_output: [batch_size, 1]

        output = self.sigmoid_act_fun(deep_output)

        return output
