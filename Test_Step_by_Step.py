import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import os
import clip
# ==================== 1. 定义VisionTokenizer模型（修正版） ====================
max_T = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VisionTokenizer(nn.Module):
    def __init__(self, d_model=512):
        super().__init__()
        # 创建ResNet-18并加载预训练权重
        resnet18 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # 去掉avgpool和fc层，保留特征提取部分
        self.backbone = nn.Sequential(*list(resnet18.children())[:-2])
        # 1x1卷积投影到目标维度
        self.conv_proj = nn.Conv2d(512, d_model, kernel_size=1)
        # 时间位置编码
        self.time_embed = nn.Embedding(max_T, d_model)

    def forward(self, images):
        """
        images: (B, T, 3, H, W)  B=批次, T=时序长度, 3=通道, H/W=图像尺寸
        return: (B, T, N, D)     N=图像Token数, D=Token维度
        """
        B, T, C, H, W = images.shape
        # 折叠时序维度到批次维度，适配ResNet输入
        x = images.view(B * T, C, H, W)

        # 提取视觉特征
        feat = self.backbone(x)  # (B*T, 512, H', W')
        feat = self.conv_proj(feat)  # (B*T, D, H', W')

        # 展平为Token序列
        D, Hp, Wp = feat.shape[1:]
        tokens = feat.flatten(2).transpose(1, 2)  # (B*T, N, D) N=Hp*Wp

        # 恢复时序维度
        tokens = tokens.view(B, T, -1, D)

        # 添加时间位置编码
        time_ids = torch.arange(T, device=images.device)
        time_emb = self.time_embed(time_ids)  # (T, D)
        time_emb = time_emb.view(1, T, 1, D)  # 广播适配
        tokens = tokens + time_emb

        return tokens


#====================== LanguageTokenize =====================

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


# ==================== 2. 图片加载与预处理函数 ====================
def load_and_preprocess_image(img_path):
    """
    加载单张图片并转换成模型要求的格式：
    1. 打开图片并转为RGB（避免灰度图/透明通道问题）
    2. 缩放、裁剪到224x224（ResNet标准输入）
    3. 归一化（符合ImageNet预训练的均值/方差）
    4. 转换成张量并调整维度为(1, 1, 3, 224, 224) → (B=1, T=1, 3, H, W)
    """
    # 定义ResNet预训练要求的预处理流程
    preprocess = transforms.Compose([
        transforms.Resize(256),  # 先缩放到256x256
        transforms.CenterCrop(224),  # 中心裁剪到224x224
        transforms.ToTensor(),  # 转为张量（0-1）
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet均值
                             std=[0.229, 0.224, 0.225])  # ImageNet方差
    ])

    # 加载图片（确保路径正确）
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"图片路径不存在：{img_path}")
    img = Image.open(img_path).convert('RGB')  # 转为RGB，避免透明通道

    # 预处理并调整维度
    img_tensor = preprocess(img)  # 形状：(3, 224, 224)
    # 扩展为(B=1, T=1, 3, 224, 224) → 适配模型输入格式
    img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)

    return img_tensor


# ==================== 3. 单张图片验证主流程 ====================
if __name__ == "__main__":
    # ===== 配置参数 =====
    img_path = "/home/xlr/outputs/captured_images/opencv__dev_video2.png"  # 替换成你的图片路径（如：./cat.jpg、/Users/xxx/photo.png）
    d_model = 512  # Token维度，和模型定义一致

    # ===== 初始化模型 =====
    tokenizer = VisionTokenizer(d_model=d_model)
    tokenizer.eval()  # 评估模式，关闭BatchNorm/Dropout

    # ===== 加载并预处理图片 =====
    try:
        input_tensor = load_and_preprocess_image(img_path)
        print(f"✅ 图片加载成功，输入张量形状：{input_tensor.shape}")
    except Exception as e:
        print(f"❌ 图片加载失败：{e}")
        exit()

    # ===== 模型推理（禁用梯度，节省内存） =====
    with torch.no_grad():
        output_tokens = tokenizer(input_tensor)

    # ===== 验证输出是否合理 =====
    print("=" * 60)
    # 1. 验证输出形状
    B, T, N, D = output_tokens.shape

    print(f"📊 输出Token形状：(B={B}, T={T}, N={N}, D={D})")

    texts = ["pick up the cube"]

    lang_encoder = LanguageTokenizer(d_model=D)

    prefixer = LanguagePrefix(num_tokens=4)

    lang_feat = lang_encoder(texts)  # (B, D)
    lang_tokens = prefixer(lang_feat)  # (B, 4, D)

    vision_tokens = torch.randn(B, T, N, D)

    full_seq = build_rt1_input_sequence(lang_tokens, vision_tokens)

    print("Language tokens:", lang_tokens.shape)
    print("Vision tokens:", vision_tokens.shape)
    print("Full sequence:", full_seq.shape)

    # # ResNet18对224x224图片下采样32倍 → 7x7=49个Token，预期N=49
    # expected_N = (224 // 32) * (224 // 32)
    # print(f"✅ 预期Token数量N：{expected_N}，实际：{N} → {'符合' if N == expected_N else '不符合'}")
    # print(f"✅ 预期Token维度D：{d_model}，实际：{D} → {'符合' if D == d_model else '不符合'}")
    #
    # # 2. 验证数值是否正常（无NaN/Inf）
    # has_nan = torch.isnan(output_tokens).any().item()
    # has_inf = torch.isinf(output_tokens).any().item()
    # print(f"❌ 输出包含NaN：{has_nan} | ❌ 输出包含Inf：{has_inf}")
    #
    # # 3. 输出Token的基础统计信息（参考）
    # print(f"📈 Token数值范围：[{output_tokens.min().item():.4f}, {output_tokens.max().item():.4f}]")
    # print(f"📈 Token均值：{output_tokens.mean().item():.4f} | 标准差：{output_tokens.std().item():.4f}")
    # print("=" * 60)
    #
    # # 可选：打印前2个Token的前5个维度值（直观查看）
    # print(f"🔍 前2个Token的前5个维度值：")
    # print(output_tokens[0, 0, :2, :5])
