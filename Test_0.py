import torch
import torch.nn as nn
import torchvision.models as models

import clip

max_T = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VisionTokenizer(nn.Module):
    def __init__(self,d_model=512):
        super().__init__()

        #创建一个ResNet-网络并使用预训练权重
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # 去掉 avgpool 和 fc
        #resnet.children()返回ResNet的所有层，[：-2]去掉最后两个（avgpool + fc），nn.Sequentioal()把剩下拼接成一个新的网络
        #ResNet的所有层：conv1 → bn → relu → maxpool → layer1 → layer2 → layer3 → layer4 → avgpool → fc
        #去掉最后两个的原因：avgpool会把空间压扁，丢失空间信息，fc会强制变成分类
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        #做卷积投影,保证所有的Token维度一样（CNN的输出维度和Transformer的维度可能不一样，不过这里是一样的）
        self.conv_proj = nn.Conv2d(512, d_model, kernel_size=1)

        # 🔑 时间 positional embedding
        self.time_embed = nn.Embedding(max_T, d_model)

    def forward(self,images):
        """
               images: (B, T, 3, H, W)
               return: (B, T, N, D)

       | 符号         | 含义                   | 直觉解释                     |
        | ---------- | -------------------- | ------------------------ |
        | **B**      | Batch size           | 一次训练用多少条轨迹               |
        | **T**      | Time steps           | 每条轨迹有多少个时间步              |
        | **C**      | Channels             | 图像通道数（RGB=3）             |
        | **H, W**   | Height, Width        | 原始图像分辨率                  |
        | **Hp, Wp** | Patch Height / Width | CNN 输出特征图的空间尺寸           |
        | **D**      | Embedding dim        | Transformer token 的维度    |
        | **N**      | Num tokens           | 每一帧图像的 token 数 = Hp × Wp |


        """
        B, T, C, H, W = images.shape
        x = images.view(B * T, C, H, W)     #ResNet不区分时序，我们把时序“乘”（并非传统意义的相乘）进去v，改变编码方式，看起来就像没有时序一样
        #“在 RT-1 的 VisionTokenizer 里，B*T 只是为了工程便利，把时间维度临时折叠成 batch，视觉模型本身完全不知道时间的存在。”
        feat = self.backbone(x)  # (B*T, 512, H', W')   3经过ResNet Backbone
        feat = self.conv_proj(feat) # (B*T, D, H', W')  投影到Transformer维度


        D, Hp, Wp = feat.shape[1:]      #Hp, Wp是下采样后的输出”维度“
        tokens = feat.flatten(2).transpose(1, 2)  # (B*T, N, D)     把2D图像变成token序列

        tokens = tokens.view(B, T, -1, D)

        # ===== 时间 positional embedding =====
        time_ids = torch.arange(T, device=images.device)  # (T,)
        time_emb = self.time_embed(time_ids)  # (T, D)
        time_emb = time_emb.view(1, T, 1, D)  # (1, T, 1, D)

        tokens = tokens + time_emb

        return tokens

class LanguageTokenizer(nn.Module):
    def __init__(self,d_model=512):
        super().__init__()

        clip_model,_ = clip.load("ViT-B/32",device=device)
        self.text_encoder = clip_model.encode_text

        self.proj = nn.Linear(512, d_model).to(device)

        #冻结CLIP
        for p in clip_model.parameters():
            p.requires_grad = False

    def forward(self, texts):
        """
        texts: list[str] of length B
        return: (B, D)
        """

        #将text的token迁移到模型所在的设备上，parameters()返回模型的所有可训练参数，next()取第一个元素（所有元素都在同一设备上，取第一个就好）
        tokens = clip.tokenize(texts).to(next(self.proj.parameters()).device)
        text_feat = self.text_encoder(tokens)   #(B,512)

        # 确保 text_feat 与 proj 权重类型一致
        text_feat = text_feat.to(self.proj.weight.dtype)

        text_feat = self.proj(text_feat)        #(B,D)

        return text_feat

class LanguagePrefix(nn.Module):
    """
        Expand language embedding into K prefix tokens
        (B,D)→(B,K,D)
    """

    def __init__(self,num_tokens=4):
        super().__init__()
        self.num_tokens = num_tokens

    def forward(self, lang_feat):
        """
            lang_feat: (B, D)
            return: (B, K, D)
        """

        lang_feat = lang_feat.to(device)

        return lang_feat.unsqueeze(1).repeat(1, self.num_tokens, 1)     #通过unsqueeze(1)新增一个序列维度，再通过repeat在这个新增维度上复制self.num_tokens次
        #所以，这个num_token是多少次呢?
        #实验结果，经验参数

def build_rt1_input_sequence(lang_tokens,vision_tokens):
    """
        lang_tokens:   (B, K, D)
        vision_tokens: (B, T, N, D)

        return:
            full_seq: (B, K + T*N, D)
    """

    B,T,N,D = vision_tokens.shape

    vision_seq = vision_tokens.view(B,T*N,D).to(device)     #改变张量shape即数据排列方式，不改变数据
    full_seq = torch.cat([lang_tokens, vision_seq], dim=1)

    return full_seq

#将动作令牌的整数编码转化为512维的向量（维度和视觉语言编码一致）
class ActionTokenizer(nn.Module):
    def __init__(self,action_vocab_size,d_model=512):
        super().__init__()
        self.action_embedding = nn.Embedding(action_vocab_size, d_model)

    def forward(self, actions_tokens):
        return self.action_embedding(actions_tokens)

def build_causal_mask(lang_len, vision_len, action_len, device):
    """

    语言 prefix tokens 和视觉 tokens 都能被任何位置看到（没时间顺序限制）

    动作 tokens 之间按时间自回归，只能看到前面的动作 token

    动作 tokens 可以看到语言 + 视觉 tokens，但语言 + 视觉 tokens 不受限制

    """
    total_len = lang_len + vision_len + action_len
    mask = torch.ones(total_len, total_len,device=device).tril()    #所有的token都能够看到自己

    mask[:lang_len + vision_len, :lang_len + vision_len] = 1       #所有的语言token和visiontkoen之间都能够相互看到

    mask[lang_len + vision_len:, :lang_len + vision_len] = 1        #aciontoken能够看到所有的langtoken和visontoken

    return mask

class RT1Transformer(nn.Module):
    def __init__(self, d_model, nhead, num_layers, action_vocab_size):
        super().__init__()
        #from torch.nn import TransformerDecoder, TransformerDecoderLayer
        from torch.nn import TransformerEncoder, TransformerEncoderLayer

        # self.transformer_layer = TransformerDecoderLayer(d_model=d_model, nhead=nhead)
        # self.transformer = TransformerDecoder(self.transformer_layer, num_layers=num_layers)
        # #num_layer的大小是如何确定的，超参数
        #
        # self.action_tokenizer = ActionTokenizer(action_vocab_size,d_model)
        # self.output_linear = nn.Linear(d_model, action_vocab_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.action_embed = nn.Embedding(action_vocab_size, d_model)
        self.head = nn.Linear(d_model, action_vocab_size)

    def forward(self, lang_tokens, vision_tokens, action_input_tokens):
        """
                lang_tokens: (B, K, D)
                vision_tokens: (B, T*N, D)
                action_input_tokens: (B, A, )  # int tokens

                returns:
                    logits over action vocab (B, A, action_vocab_size)      A是单个轨迹动作token的个数
        """
        # B = lang_tokens.size(0)
        # device = lang_tokens.device
        #
        # action_emb = self.action_tokenizer(action_input_tokens)     #(B,A,D)

        # #拼接所有token
        # src = torch.cat([lang_tokens, vision_tokens, action_emb], dim=1)
        #
        # #构造mask
        # lang_len = lang_tokens.size(1)
        # vision_len = vision_tokens.size(1)
        # action_len = action_input_tokens.size(1)
        # mask = build_causal_mask(lang_len, vision_len, action_len, device)
        #
        # # transformer expects mask with False where attend allowed, True where blocked
        # attn_mask = ~mask.bool()    #转化为transformer允许的形式，把元矩阵中的1变换为True,0toFalse
        #
        # src = src.transpose(0, 1)       #交换Batch和拼接后的Token长度的位置（交换后才符合Transformer的要求）
        # output = self.transformer(tgt=src, memory=None, tgt_mask=attn_mask)
        #
        # output = output.transpose(0, 1)
        # logits = self.output_linear(output[:, lang_len+vision_len:, :])
        #
        # return logits

        B = lang_tokens.size(0)
        device = lang_tokens.device

        action_emb = self.action_embed(action_input_tokens)

        src = torch.cat([lang_tokens, vision_tokens, action_emb], dim=1)

        lang_len = lang_tokens.size(1)
        vision_len = vision_tokens.size(1)
        action_len = action_input_tokens.size(1)

        attn_mask = build_causal_mask(lang_len, vision_len, action_len, device)

        out = self.transformer(src, mask=attn_mask)

        logits = self.head(out[:, lang_len + vision_len:])

        return logits


def sanity_forward_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B = 2
    T = 4
    H = W = 224
    D = 512
    N = 49
    K = 4
    A = 6
    action_vocab = 20

    images = torch.randn(B, T, 3, H, W).to(device)
    texts = ["pick up cube", "grasp object"]
    actions = torch.randint(0, action_vocab, (B, A)).to(device)

    vision = VisionTokenizer(d_model=D).to(device)
    lang = LanguageTokenizer(d_model=D).to(device)
    prefix = LanguagePrefix(K).to(device)
    policy = RT1Transformer(D, 8, 4, action_vocab).to(device)

    with torch.no_grad():
        v_tokens = vision(images)            # (B,T,N,D)
        l_feat = lang(texts)                 # (B,D)
        l_tokens = prefix(l_feat)            # (B,K,D)
        v_seq = v_tokens.reshape(B, T*N, D)
        logits = policy(l_tokens, v_seq, actions)

    print("logits:", logits.shape)

if __name__ == "__main__":
    sanity_forward_test()