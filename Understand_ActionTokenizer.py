import torch
import torch.nn as nn


class ActionTokenizer(nn.Module):
    def __init__(self, action_vocab_size, d_model):
        super().__init__()
        self.action_embedding = nn.Embedding(action_vocab_size, d_model)

    def forward(self, action_tokens):
        return self.action_embedding(action_tokens)  # (B, max_action_len, D)


# 测试：模拟机器人动作场景
action_vocab_size = 8  # 假设有8种基础动作（0~7）
d_model = 512  # 嵌入维度和视觉/语言Token一致
action_tokenizer = ActionTokenizer(action_vocab_size, d_model)

# 模拟输入：B=2个样本，每个样本的动作序列长度max_action_len=4
# 动作令牌是整数：比如[0=前进, 1=后退, 4=抓取, 7=放下]
action_tokens = torch.tensor([
    [0, 4, 7, 0],  # 样本1：前进→抓取→放下→前进
    [1, 4, 7, 1]  # 样本2：后退→抓取→放下→后退
])

# 前向传播
action_embeds = action_tokenizer(action_tokens)

# 查看维度
print(f"输入动作令牌形状：{action_tokens.shape}")  # torch.Size([2, 4])
print(f"输出动作嵌入形状：{action_embeds.shape}")  # torch.Size([2, 4, 512])

# 验证：语义相似的动作嵌入向量更接近（训练后）
# 这里是随机初始化，训练后前进(0)和后退(1)的向量余弦相似度会更高
sim_0_1 = torch.cosine_similarity(action_embeds[0, 0], action_embeds[1, 0], dim=0)
sim_0_4 = torch.cosine_similarity(action_embeds[0, 0], action_embeds[0, 1], dim=0)
print(f"前进(0)和后退(1)的相似度：{sim_0_1.item():.4f}")
print(f"前进(0)和抓取(4)的相似度：{sim_0_4.item():.4f}")
