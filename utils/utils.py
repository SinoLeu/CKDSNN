import cv2
import numpy as np
import torch
import torch.nn.functional as F
# 设置随机种子
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical, kl_divergence
import torch
import torch.nn.functional as F
import torch.nn as nn



def cc_loss(logits_student, logits_teacher, temperature, reduce=True):
    batch_size, class_num = logits_teacher.shape
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    student_matrix = torch.mm(pred_student.transpose(1, 0), pred_student)
    teacher_matrix = torch.mm(pred_teacher.transpose(1, 0), pred_teacher)
    if reduce:
        consistency_loss = ((teacher_matrix - student_matrix) ** 2).sum() / class_num
    else:
        consistency_loss = ((teacher_matrix - student_matrix) ** 2) / class_num
    return consistency_loss

def bc_loss(logits_student, logits_teacher, temperature, reduce=True):
    batch_size, class_num = logits_teacher.shape
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    student_matrix = torch.mm(pred_student, pred_student.transpose(1, 0))
    teacher_matrix = torch.mm(pred_teacher, pred_teacher.transpose(1, 0))
    if reduce:
        consistency_loss = ((teacher_matrix - student_matrix) ** 2).sum() / batch_size
    else:
        consistency_loss = ((teacher_matrix - student_matrix) ** 2) / batch_size
    return consistency_loss

def gram_matrix_logits(logits):
  """
  计算 logits 的 Gram 矩阵。
  输入:
    logits: 张量，形状为 (b, c)
  输出:
    gram: 张量，形状为 (b, c, c)
  """
  b, c = logits.size()
  # 将 logits reshape 为 (b, c, 1) 的 "特征图"
  features = logits.view(b, c, 1)
  # 计算 Gram 矩阵
  gram = torch.bmm(features, features.transpose(1, 2))
  return gram

def logits_external_loss(logits_student, logits_teacher, temperature=2.0, reduce=True):
    
    return bc_loss(logits_student, logits_teacher, temperature,reduce) + cc_loss(logits_student, logits_teacher, temperature,reduce)
## class internal loss
def logits_internal_loss(student_logits, teacher_logits, temperature=2.0, reduce=True):
  """
  计算学生和教师 logits 的 Gram 矩阵之间的均方误差损失。
  输入:
    student_logits: 学生模型的 logits，形状为 (b, c)
    teacher_logits: 教师模型的 logits，形状为 (b, c)
    temperature: 温度系数 (可选)
  输出:
    loss: 标量，Gram 矩阵损失
  """
  batch_size, class_num = teacher_logits.shape
  # 选项 1: 在计算 Gram 矩阵之前应用温度系数
#   student_logits_temp = student_logits / temperature
#   teacher_logits_temp = teacher_logits / temperature
  pred_student = F.softmax(student_logits / temperature, dim=1)
  pred_teacher = F.softmax(teacher_logits / temperature, dim=1)

  student_matrix = gram_matrix_logits(pred_student)
  teacher_matrix = gram_matrix_logits(pred_teacher)

  # 选项 2: 直接对原始 logits 计算 Gram 矩阵
  # student_gram = gram_matrix_logits(student_logits)
  # teacher_gram = gram_matrix_logits(teacher_logits)
  if reduce:
    consistency_loss = ((teacher_matrix - student_matrix) ** 2).sum() / batch_size
  else:
    consistency_loss = ((teacher_matrix - student_matrix) ** 2) / batch_size
  return consistency_loss

def pearson_correlation_kd(logits_teacher, logits_student):
    """
    向量化计算logits之间的皮尔逊相关系数损失
    
    参数:
        logits_teacher: (B, C) 教师模型的logits
        logits_student: (B, C) 学生模型的logits
    
    返回:
        平均皮尔逊相关性损失 (标量 Tensor)
    """
    # 计算每个类别的均值 (形状: (C,))
    mean_t = torch.mean(logits_teacher, dim=0)  # (C,)
    mean_s = torch.mean(logits_student, dim=0)  # (C,)
    
    # 中心化logits (减去均值)
    centered_teacher = logits_teacher - mean_t  # (B, C)
    centered_student = logits_student - mean_s  # (B, C)
    
    # 计算协方差 (形状: (C,))
    cov = torch.sum(centered_teacher * centered_student, dim=0)  # (C,)
    
    # 计算标准差 (形状: (C,))
    std_t = torch.sqrt(torch.sum(centered_teacher ** 2, dim=0))  # (C,)
    std_s = torch.sqrt(torch.sum(centered_student ** 2, dim=0))  # (C,)
    
    # 防止除零错误，计算皮尔逊相关系数 (形状: (C,))
    r = cov / (std_t * std_s + 1e-8)
    
    # 计算损失 (1 - r)，并取平均
    correlation_loss = torch.mean(1 - r)
    
    return correlation_loss

def apply_adaptive_noise(logits):

    mean = logits.mean(dim=-1, keepdims=True)
    stdv = logits.std(dim=-1, keepdims=True)

    # 使用噪音模拟时间步骤，
    logits_noisy = [
        logits,
        logits + torch.randn_like(logits) * mean + stdv,
        logits + torch.randn_like(logits) * mean + stdv,
        logits + torch.randn_like(logits) * mean + stdv  
    ]
    
    logits_averaged = torch.stack(logits_noisy, dim=0).mean(0)
    return logits_averaged

def apply_adaptive_flatten(flatten,noise_type = "dropout"):
    
    if noise_type == "gauss":
        return apply_adaptive_flatten_gauss(flatten)
    elif noise_type == "mix":
        return apply_adaptive_flatten_mix(flatten)
    elif noise_type == "dropout":
        return apply_dropout(flatten)

def apply_dropout(flatten, p=[0.1,0.1,0.2,0.2,0.2,0.3,0.3,0.3,0.4]):
    flatten_noisy = []

    # 对每个 p_i 应用 dropout
    for p_i in p:
        if not 0 <= p_i < 1:
            print(f"Warning: Invalid dropout probability {p_i}, skipping")
            continue
        dropout = nn.Dropout(p=p_i).to(flatten.device)
        dropout.train()  # 训练模式，确保 dropout 生效
        noisy_flatten = dropout(flatten)
        noisy_flatten = torch.clamp(noisy_flatten, min=-100, max=100)
        flatten_noisy.append(noisy_flatten)
    
    flatten_averaged = torch.stack(flatten_noisy, dim=0).mean(dim=0)
    
    return flatten_averaged

def apply_adaptive_flatten_mix(flatten):
    alpha=0.5
    T=10
    """
    基于 Mixup 扰动生成虚拟样本并平均。
    
    参数:
        flatten: 输入张量，形状 (batch_size, feature_dim)
        alpha: Beta 分布参数，控制混合强度
        T: 扰动版本数量（包括原始 flatten）
    """
    # 检查输入
    if torch.isnan(flatten).any() or torch.isinf(flatten).any():
        print("Warning: flatten contains NaN or Inf, returning original flatten")
        return flatten
    flatten = torch.clamp(flatten, min=-100, max=100)

    batch_size = flatten.size(0)
    flatten_noisy = []

    # 添加原始 flatten
    flatten_noisy.append(flatten)

    # 循环生成 T-1 个 Mixup 扰动版本
    for _ in range(T-1):
        # 生成插值系数 lambda
        lambda_ = torch.distributions.beta.Beta(alpha, alpha).sample((batch_size, 1)).to(flatten.device)
        
        # 随机打乱索引，选择另一个样本
        indices = torch.randperm(batch_size).to(flatten.device)
        
        # Mixup 插值
        noisy_flatten = lambda_ * flatten + (1 - lambda_) * flatten[indices]
        
        # 裁剪防止溢出
        noisy_flatten = torch.clamp(noisy_flatten, min=-100, max=100)
        
        flatten_noisy.append(noisy_flatten)
    
    # 堆叠并平均
    flatten_averaged = torch.stack(flatten_noisy, dim=0).mean(0)
    flatten_averaged = torch.clamp(flatten_averaged, min=-100, max=100)
    
    # 调试信息
    # print("flatten_averaged max:", flatten_averaged.max().item(), "min:", flatten_averaged.min().item())
    # print("flatten_averaged has NaN:", torch.isnan(flatten_averaged).any().item())
    # print("flatten_averaged has Inf:", torch.isinf(flatten_averaged).any().item())
    
    return flatten_averaged

def apply_adaptive_flatten_gauss(flatten):

    mean = flatten.mean(dim=-1, keepdims=True)
    stdv = flatten.std(dim=-1, keepdims=True)

    # T = torch.randint(low=1, high=30, size=(1,)).item()
    T = 10
    noise = torch.randn(T-1, *flatten.shape, device=flatten.device) * mean + stdv
    flatten_noisy = torch.cat([flatten.unsqueeze(0), flatten.unsqueeze(0) + noise], dim=0)
    
    flatten_averaged = flatten_noisy.mean(0)
    return flatten_averaged


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def cam_to_prob(cam):
    """
    将 CAM 图转换为概率分布。

    参数：
    - cam (Tensor): 需要转换的 CAM 图 (B, H, W)。

    返回：
    - prob_map (Tensor): 转换后的概率图 (B, H, W)。
    """
    # 对 CAM 图应用 Softmax 转换为概率图，沿着 H 和 W 维度进行归一化
    prob_map = F.softmax(cam.view(cam.size(0), -1), dim=1).view_as(cam)
    return prob_map


def cam_to_zscore(cam, eps=1e-8):
    """
    对 CAM 图进行 Z-score 标准化，支持 3D 或 4D 输入。

    参数:
    - cam (Tensor): CAM 图，shape 可为 [B, H, W] 或 [B, C, H, W]
    - eps (float): 防止除零的小常数

    返回:
    - zscore_map (Tensor): 标准化后的 CAM 图，与输入 shape 一致
    """
    original_shape = cam.shape

    # 自动判断是否为 3D 或 4D 输入
    if cam.dim() == 3:
        # Case 1: 3D Tensor -> [B, H, W]
        B, H, W = cam.shape
        cam_flat = cam.view(B, -1)  # [B, H*W]
    elif cam.dim() == 4:
        # Case 2: 4D Tensor -> [B, C, H, W]
        B, C, H, W = cam.shape
        cam_flat = cam.view(B, -1)  # [B, C*H*W]，但只取空间维度
    else:
        raise ValueError(f"Unsupported input dimension: {cam.dim()}")

    # 计算均值和标准差
    mean = cam_flat.mean(dim=1, keepdim=True)  # [B, 1]
    std = cam_flat.std(dim=1, keepdim=True)    # [B, 1]

    # Z-score 标准化
    cam_flat = (cam_flat - mean) / (std + eps)

    # 恢复原始形状
    cam_normalized = cam_flat.view(original_shape)

    return cam_normalized

def pearson_correlation_loss(x, y):
    """
    向量化计算批量 2D 张量的皮尔逊相关系数损失，平均每个样本的 1 - r
    参数:
        x: 3D 张量，形状 (batch_size, m, n)，预测值
        y: 3D 张量，形状 (batch_size, m, n)，真实值
    返回:
        平均损失值 (1 - 平均皮尔逊相关系数)
    """
    # 确保输入是 3D 张量
    assert x.dim() == 3 and y.dim() == 3, "输入必须是 3D 张量 (batch_size, m, n)"
    assert x.size() == y.size(), "两个张量的形状必须相同"
    
    batch_size = x.size(0)
    
    # 展平每个样本的 2D 张量为 1D：(batch_size, m, n) -> (batch_size, m*n)
    x_flat = x.view(batch_size, -1)
    y_flat = y.view(batch_size, -1)
    
    # 计算均值：(batch_size,)
    mean_x = torch.mean(x_flat.float(), dim=1, keepdim=True)
    mean_y = torch.mean(y_flat.float(), dim=1, keepdim=True)
    
    # 去中心化
    xm = x_flat - mean_x
    ym = y_flat - mean_y
    
    # 计算协方差和标准差
    cov_xy = torch.sum(xm * ym, dim=1)  # (batch_size,)
    std_x = torch.sqrt(torch.sum(xm ** 2, dim=1))  # (batch_size,)
    std_y = torch.sqrt(torch.sum(ym ** 2, dim=1))  # (batch_size,)
    
    # 防止除以零：如果 std_x 或 std_y 为零，设置 r = 0（损失为 1）
    mask = (std_x != 0) & (std_y != 0)
    r = torch.zeros_like(cov_xy)
    r[mask] = cov_xy[mask] / (std_x[mask] * std_y[mask])
    
    # 损失：1 - r
    loss = 1.0 - r
    
    # 返回平均损失
    return torch.mean(loss)

def compute_kl_divergence_zscore(preds_S, preds_T, eps=1e-10,Tau=2.0):
    """
    计算学生模型 (preds_S) 和教师模型 (preds_T) 的 KL 散度。

    参数：
    - preds_S (Tensor): 学生模型的 CAM 图，大小为 (B, H, W)
    - preds_T (Tensor): 教师模型的 CAM 图，大小为 (B, H, W)
    - eps (float): 避免除以零的小常数。

    返回：
    - kl_loss (Tensor): KL 散度损失。
    """
    # 将 CAM 图转换为概率图
    prob_S = cam_to_zscore(preds_S/Tau)
    prob_T = cam_to_zscore(preds_T/Tau)
    
    # 防止零值对数的影响
    prob_S = prob_S + eps
    prob_T = prob_T + eps
    
    # 计算 KL 散度
    # kl_loss = torch.sum(prob_S * torch.log(prob_S / prob_T), dim=(1, 2)).mean()  # 对空间维度H, W求和，并取平均
    kl_loss = F.kl_div(prob_S.log(), prob_T, reduction='batchmean')
    # pearson_loss = pearson_correlation_loss(prob_S, prob_T)
    return kl_loss
def compute_kl_divergence(preds_S, preds_T, eps=1e-10,Tau=2.0):
    """
    计算学生模型 (preds_S) 和教师模型 (preds_T) 的 KL 散度。

    参数：
    - preds_S (Tensor): 学生模型的 CAM 图，大小为 (B, H, W)
    - preds_T (Tensor): 教师模型的 CAM 图，大小为 (B, H, W)
    - eps (float): 避免除以零的小常数。

    返回：
    - kl_loss (Tensor): KL 散度损失。
    """
    # 将 CAM 图转换为概率图
    prob_S = cam_to_prob(preds_S/Tau)
    prob_T = cam_to_prob(preds_T/Tau)
    
    # 防止零值对数的影响
    prob_S = prob_S + eps
    prob_T = prob_T + eps
    
    # 计算 KL 散度
    # kl_loss = torch.sum(prob_S * torch.log(prob_S / prob_T), dim=(1, 2)).mean()  # 对空间维度H, W求和，并取平均
    kl_loss = F.kl_div(prob_S.log(), prob_T, reduction='batchmean')
    # pearson_loss = pearson_correlation_loss(prob_S, prob_T)
    return kl_loss


def freeze_model_parameters(model):
    """
    Freeze all parameters in the model to disable gradient computation.

    Args:
        model (nn.Module): The neural network model to freeze.
    """
    for param in model.parameters():
        param.requires_grad = False

def mmd_loss(source_logits, target_logits, sigma=1.0):
    
    assert source_logits.size(1) == target_logits.size(1), "源域和目标域的类别数必须相同"
    
    n_s = source_logits.size(0)  # 源域样本数
    n_t = target_logits.size(0)  # 目标域样本数
    
    # 计算源域内、目标域内和跨域的核矩阵
    # 使用 torch.cdist 计算欧氏距离矩阵
    dist_ss = torch.cdist(source_logits, source_logits, p=2) ** 2  # 源域样本间的距离平方
    dist_tt = torch.cdist(target_logits, target_logits, p=2) ** 2  # 目标域样本间的距离平方
    dist_st = torch.cdist(source_logits, target_logits, p=2) ** 2  # 源域与目标域样本间的距离平方
    
    # 计算高斯核
    kernel_ss = torch.exp(-dist_ss / (2 * sigma ** 2))
    kernel_tt = torch.exp(-dist_tt / (2 * sigma ** 2))
    kernel_st = torch.exp(-dist_st / (2 * sigma ** 2))
    
    # 计算 MMD 的三项
    term_ss = kernel_ss.sum() / (n_s * n_s)  # 源域内核均值
    term_tt = kernel_tt.sum() / (n_t * n_t)  # 目标域内核均值
    term_st = kernel_st.sum() / (n_s * n_t)  # 跨域核均值
    
    # 完整 MMD 损失
    mmd = term_ss + term_tt - 2 * term_st
    
    return mmd

def entropy_loss(logits_stu, temperature=1.0):
    """
    计算学生模型 logits 的熵。
    
    参数：
        logits_stu: 学生模型的 logits，形状为 (batch_size, num_classes)
        temperature: 温度参数，用于平滑 softmax 分布（默认 1.0）
    
    返回：
        entropy: 熵值（标量，batch 维度的平均值）
    """
    # 应用温度缩放
    logits_stu = logits_stu / temperature
    
    # 计算 softmax 概率
    prob_stu = F.softmax(logits_stu, dim=1)
    
    # 计算 log softmax（提高数值稳定性）
    log_prob_stu = F.log_softmax(logits_stu, dim=1)
    
    # 计算熵：-sum(p * log(p))
    entropy = -torch.sum(prob_stu * log_prob_stu, dim=1).mean()
    
    return entropy

def cross_entropy_loss(logits_tea, logits_stu, temperature=2.0):
    """
    计算教师和学生模型 logits 之间的交叉熵损失（知识蒸馏）。
    
    参数：
        logits_tea: 教师模型的 logits，形状为 (batch_size, num_classes)
        logits_stu: 学生模型的 logits，形状为 (batch_size, num_classes)
        temperature: 温度参数，用于平滑 softmax 分布
    
    返回：
        loss: 交叉熵损失（标量）
    """
    # 应用温度缩放
    logits_tea = logits_tea / temperature
    logits_stu = logits_stu / temperature
    
    # 计算 softmax 概率（教师的软标签）
    prob_tea = F.softmax(logits_tea, dim=1)
    
    # 计算 log softmax（学生预测的 log 概率）
    log_prob_stu = F.log_softmax(logits_stu, dim=1)
    
    # 计算交叉熵损失
    loss = -torch.sum(prob_tea * log_prob_stu, dim=1).mean()
    
    # 乘以 temperature^2 调整量级
    loss = loss * (temperature ** 2)
    
    return loss

def soft_kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 2.0,
    reduction: str = 'mean'
) -> torch.Tensor:
    # 对logits应用温度缩放
    student_scaled = student_logits / temperature
    teacher_scaled = teacher_logits / temperature
    
    # 计算软化的概率分布
    student_soft = F.softmax(student_scaled, dim=1)
    teacher_soft = F.softmax(teacher_scaled, dim=1)
    
    # 计算log概率，用于KL散度
    student_log_soft = F.log_softmax(student_scaled, dim=1)
    
    # 计算KL散度
    # 注意：我们需要乘以temperature^2来调整损失的尺度
    kd_loss = F.kl_div(
        student_log_soft,
        teacher_soft,
        reduction=reduction
    ) * (temperature ** 2)
    
    return kd_loss


def generate_gaussian_noise(logits,device):
    """
    根据 logit 计算的均值和标准差生成符合该分布的高斯噪声
    :param logit: 形状为 (..., N) 的 torch.Tensor
    :return: 具有相同形状的高斯噪声
    """
    # w = 0.1
    # 计算 logit 的均值和标准差
    mean = logits.mean(dim=-1, keepdim=True)
    stdv = logits.std(dim=-1, keepdim=True)  # 

    # 生成标准正态噪声（均值 0，标准差 1）
    noise = torch.randn_like(logits).to(device)

    # 调整噪声的均值和标准差，使其匹配 logit 的分布
    gaussian_noise = mean + noise * (stdv**2)

    return gaussian_noise


def generate_gaussian_noise_weight(logits,weight=0.5):
    device = logits.device
    """
    根据 logit 计算的均值和标准差生成符合该分布的高斯噪声
    :param logit: 形状为 (..., N) 的 torch.Tensor
    :return: 具有相同形状的高斯噪声
    """
    # w = 0.1
    # 计算 logit 的均值和标准差
    mean = logits.mean(dim=-1, keepdim=True)
    stdv = logits.std(dim=-1, keepdim=True)  # 计算总体标准差

    # 生成标准正态噪声（均值 0，标准差 1）
    noise = torch.randn_like(logits).to(device)

    # 调整噪声的均值和标准差，使其匹配 logit 的分布
    gaussian_noise = mean + noise * (stdv**2)

    return logits+weight*gaussian_noise

def smooth_noise_softmax(logits, smooth_temp=2.0, noise_temp=5.0, weight=0.90):
    """
    将 Softmax 平滑和指定的噪声类型结合用于生成平滑而具有区分性的分布。

    参数:
    - logits (torch.Tensor): 原始 logits 输入。
    - smooth_temp (float): 用于 Softmax 平滑的温度参数，较大值可增强平滑效果。
    - noise_type (str): 噪声类型，支持 "gumbel", "gaussian", "uniform", "laplace", "poisson"。
    - noise_temp (float): 噪声温度参数，控制噪声的强度。
    - weight (float): 平滑分布和噪声分布的权重比例，介于 0 和 1 之间。

    返回:
    - torch.Tensor: 经过平滑和噪声处理后的最终分布。
    """
    # 获取 logits 的设备
    device = logits.device

    # 1. Softmax 平滑处理
    smoothed_probs = F.softmax(logits / smooth_temp, dim=1)
    noisy_logits = generate_gaussian_noise(logits,device=device) 
    noisy_probs = F.softmax(noisy_logits / noise_temp, dim=1)

    # 3. 组合并归一化
    # combined_probs = smoothed_probs
    ## noisy_probs = smoothed_probs
    combined_probs = weight * smoothed_probs + (1 - weight) * noisy_probs

    return combined_probs

#     return kl_loss

def compute_entropy(prob, is_temporal=False):
    """
    计算输出 logits 的熵。
    
    参数：
        logits: Tensor，形状为 [batch_size, num_classes]（非时序）或 [batch_size, timesteps, num_classes]（时序）。
        is_temporal: bool，是否为时序数据（SNN）。
    
    返回：
        entropy: 标量，平均熵值。
    """
    # 转换为概率分布
    # prob = torch.softmax(logits, dim=-1)  # 沿类别维度 softmax
    
    # 避免 log(0)，添加小常数
    prob = prob + 1e-10
    
    # 计算熵：-sum(p * log(p))
    entropy = -torch.sum(prob * torch.log(prob), dim=-1)  # 形状为 [batch_size] 或 [batch_size, timesteps]
    
    if is_temporal:
        # 时序数据：对时间步和批量取平均
        entropy = entropy.mean(dim=[0, 1])  # 平均熵，标量
    else:
        # 非时序数据：对批量取平均
        entropy = entropy.mean(dim=0)  # 平均熵，标量
    
    return entropy.item()

def soft_loss_smooth(student_logits,teacher_logits, temperature=2.0,noise_weight=0.99):
    # 对教师模型输出进行温度缩放后的softmax
    soft_targets = F.softmax(teacher_logits / temperature, dim=1)
    
    # 学生模型输出也应用相同的温度缩放和平滑处理
    soft_student_outputs = smooth_noise_softmax(student_logits,weight=noise_weight, smooth_temp=temperature, noise_temp=temperature)
    
    # 计算交叉熵损失
    kl_loss = F.kl_div(soft_student_outputs.log(), soft_targets, reduction='batchmean') * (temperature ** 2)
    
    return kl_loss,compute_entropy(soft_student_outputs)


def _get_gt_mask(logits, target):
    """生成目标类别掩码"""
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1.0)
    return mask

def _get_other_mask(logits, target):
    """生成非目标类别掩码"""
    target = target.reshape(-1)
    mask = torch.ones_like(logits)
    mask = mask.scatter_(1, target.unsqueeze(1), 0.0)
    return mask

def cat_mask(tensor, gt_mask, other_mask):
    """
    应用掩码到张量，保留目标和非目标类别的值
    假设：将 gt_mask 和 other_mask 应用后，生成新的张量
    """
    return tensor * (gt_mask + other_mask)

def top_k_dkd_loss(logits_student, logits_teacher, target, alpha=1.0, beta=4.0, temperature=2.0, top_k=5):
    """
    DKD 损失函数，在 NCKD 部分仅蒸馏非目标类别中的 top-k 概率
    参数:
        logits_student: 学生模型的 logits，形状 (batch_size, num_classes)
        logits_teacher: 教师模型的 logits，形状 (batch_size, num_classes)
        target: 真实标签，形状 (batch_size,)
        alpha: TCKD 损失的权重
        beta: NCKD 损失的权重
        temperature: 温度参数
        top_k: NCKD 中蒸馏的非目标类别概率数量
    返回:
        总损失：alpha * tckd_loss + beta * nckd_loss
    """
    # 确保输入形状正确
    assert logits_student.dim() == 2 and logits_teacher.dim() == 2, "logits 必须是 2D 张量"
    assert logits_student.size() == logits_teacher.size(), "学生和教师 logits 形状必须相同"
    assert target.dim() == 1, "target 必须是 1D 张量"
    
    batch_size, num_classes = logits_student.size()
    
    # 生成目标和非目标类别掩码
    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)
    
    # 计算软化概率分布
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    
    # TCKD：目标类别知识蒸馏
    pred_student_tckd = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher_tckd = cat_mask(pred_teacher, gt_mask, other_mask)
    log_pred_student_tckd = torch.log(pred_student_tckd)
    tckd_loss = (
        F.kl_div(log_pred_student_tckd, pred_teacher_tckd, reduction='sum')
        * (temperature**2)
        / batch_size
    )
    
    # NCKD：非目标类别知识蒸馏，仅蒸馏 top-k 概率
    # 抑制目标类别
    logits_teacher_nckd = logits_teacher / temperature - 1000.0 * gt_mask
    logits_student_nckd = logits_student / temperature - 1000.0 * gt_mask
    
    # 计算非目标类别的软化概率
    pred_teacher_nckd = F.softmax(logits_teacher_nckd, dim=1)
    pred_student_nckd = F.softmax(logits_student_nckd, dim=1)
    
    # 筛选非目标类别中的 top-k 概率
    # 首先屏蔽目标类别
    pred_teacher_nckd_masked = pred_teacher_nckd * other_mask
    # 获取 top-k 概率的索引
    _, topk_indices = torch.topk(pred_teacher_nckd_masked, k=top_k, dim=1)
    # 创建 top-k 掩码
    topk_mask = torch.zeros_like(pred_teacher_nckd).scatter_(1, topk_indices, 1.0)
    
    # 仅保留 top-k 非目标类别的概率
    pred_teacher_nckd_topk = pred_teacher_nckd * topk_mask
    pred_student_nckd_topk = pred_student_nckd * topk_mask
    
    # 归一化 top-k 概率（确保概率和为 1）
    pred_teacher_nckd_topk = pred_teacher_nckd_topk / (pred_teacher_nckd_topk.sum(dim=1, keepdim=True) + 1e-10)
    pred_student_nckd_topk = pred_student_nckd_topk / (pred_student_nckd_topk.sum(dim=1, keepdim=True) + 1e-10)
    
    # 计算 top-k 非目标类别的 log 概率
    log_pred_student_nckd_topk = torch.log(pred_student_nckd_topk + 1e-10)  # 防止 log(0)
    
    # NCKD 损失：仅对 top-k 类别计算 KL 散度
    nckd_loss = (
        F.kl_div(log_pred_student_nckd_topk, pred_teacher_nckd_topk, reduction='sum')
        * (temperature**2)
        / batch_size
    )
    
    # 总损失
    return alpha * tckd_loss + beta * nckd_loss

# def soft_loss_smooth(ann_output, outputs, Tau, noise_type="gumbel"):
#     # 对教师模型输出进行温度缩放后的softmax
#     soft_targets = F.softmax(ann_output / Tau, dim=1)
    
#     # 学生模型输出也应用相同的温度缩放和平滑处理
#     soft_student_outputs = smooth_noise_softmax(outputs, smooth_temp=Tau, noise_type=noise_type, noise_temp=Tau)
    
#     # 计算交叉熵损失
#     kl_loss = F.kl_div(soft_student_outputs.log(), soft_targets, reduction='batchmean') * (Tau ** 2)
    
#     return kl_loss

# def smooth_noise_softmax(logits, smooth_temp=5, noise_type="gumbel", noise_temp=5.0, weight=0.1):
#     """
#     将 Softmax 平滑和指定的噪声类型结合用于生成平滑而具有区分性的分布。

#     参数:
#     - logits (torch.Tensor): 原始 logits 输入。
#     - smooth_temp (float): 用于 Softmax 平滑的温度参数，较大值可增强平滑效果。
#     - noise_type (str): 噪声类型，支持 "gumbel", "gaussian", "uniform", "laplace", "poisson"。
#     - noise_temp (float): 噪声温度参数，控制噪声的强度。
#     - weight (float): 平滑分布和噪声分布的权重比例，介于 0 和 1 之间。

#     返回:
#     - torch.Tensor: 经过平滑和噪声处理后的最终分布。
#     """
#     # 获取 logits 的设备
#     device = logits.device

#     # 1. Softmax 平滑处理
#     smoothed_probs = F.softmax(logits / smooth_temp, dim=1)

#     # noisy_probs = F.gumbel_softmax(logits, tau=noise_temp, hard=False)
#     gaussian_noise = torch.randn_like(logits) * noise_temp
#     gaussian_noise = gaussian_noise.to(device)
#     noisy_logits = logits + gaussian_noise
#     noisy_probs = F.softmax(noisy_logits / smooth_temp, dim=1)
#     # 2. 添加噪声处理
#     # if noise_type == "gumbel":
#     #     noisy_probs = F.gumbel_softmax(logits, tau=noise_temp, hard=False)
#     # elif noise_type == "gaussian":
#     #     gaussian_noise = torch.randn_like(logits) * noise_temp
#     #     # 确保噪声在正确的设备上
#     #     gaussian_noise = gaussian_noise.to(device)
#     #     noisy_logits = logits + gaussian_noise
#     #     noisy_probs = F.softmax(noisy_logits / smooth_temp, dim=1)
#     # elif noise_type == "uniform":
#     #     uniform_noise = torch.rand_like(logits) * noise_temp
#     #     # 确保噪声在正确的设备上
#     #     uniform_noise = uniform_noise.to(device)
#     #     noisy_logits = logits + uniform_noise
#     #     noisy_probs = F.softmax(noisy_logits / smooth_temp, dim=1)
#     # elif noise_type == "laplace":
#     #     laplace_noise = torch.distributions.Laplace(0., noise_temp).sample(logits.shape).to(device)
#     #     noisy_logits = logits + laplace_noise
#     #     noisy_probs = F.softmax(noisy_logits / smooth_temp, dim=1)
#     # elif noise_type == "poisson":
#     #     poisson_noise = torch.poisson(torch.ones_like(logits) * noise_temp).to(device)
#     #     noisy_logits = logits + poisson_noise
#     #     noisy_probs = F.softmax(noisy_logits / smooth_temp, dim=1)
#     # else:
#     #     raise ValueError(f"Unsupported noise type: {noise_type}")

#     # 3. 组合并归一化
#     combined_probs = weight * smoothed_probs + (1 - weight) * noisy_probs

#     return combined_probs


def pearson_kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 2.0,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    基于皮尔逊相关系数的知识蒸馏损失函数
    参数:
        student_logits: 学生模型的logits，形状 (batch_size, num_classes)
        teacher_logits: 教师模型的logits，形状 (batch_size, num_classes)
        temperature: 温度参数，用于缩放logits（保留以兼容原始接口）
        reduction: 损失聚合方式，'mean' 或 'sum'
    返回:
        皮尔逊相关系数损失（1 - r 的聚合值）
    """
    # 确保输入是 2D 张量
    assert student_logits.dim() == 2 and teacher_logits.dim() == 2, "输入必须是 2D 张量 (batch_size, num_classes)"
    assert student_logits.size() == teacher_logits.size(), "两个张量的形状必须相同"

    batch_size = student_logits.size(0)
    
    # 可选：应用温度缩放（保留以兼容原始接口）
    student_scaled = student_logits / temperature
    teacher_scaled = teacher_logits / temperature
    
    # 计算均值：(batch_size, 1)
    mean_student = torch.mean(student_scaled.float(), dim=1, keepdim=True)
    mean_teacher = torch.mean(teacher_scaled.float(), dim=1, keepdim=True)
    
    # 去中心化
    student_centered = student_scaled - mean_student
    teacher_centered = teacher_scaled - mean_teacher
    
    # 计算协方差和标准差
    cov = torch.sum(student_centered * teacher_centered, dim=1)  # (batch_size,)
    std_student = torch.sqrt(torch.sum(student_centered ** 2, dim=1))  # (batch_size,)
    std_teacher = torch.sqrt(torch.sum(teacher_centered ** 2, dim=1))  # (batch_size,)
    
    # 防止除以零：如果 std_student 或 std_teacher 为零，设置 r = 0（损失为 1）
    mask = (std_student != 0) & (std_teacher != 0)
    r = torch.zeros_like(cov)
    r[mask] = cov[mask] / (std_student[mask] * std_teacher[mask])
    
    # 损失：1 - r
    loss = 1.0 - r  # (batch_size,)
    
    # 根据 reduction 参数聚合损失
    if reduction == 'mean':
        return torch.mean(loss)
    elif reduction == 'sum':
        return torch.sum(loss)
    else:
        raise ValueError("reduction 必须是 'mean' 或 'sum'")
    
def mean_CAM(feature_conv):
    cam = feature_conv.mean(dim=0).mean(axis=1)  # Shape: (H, W)
    
    # 激活值归一化处理
    cam = cam - cam.min()  # 确保非负
    cam_img = cam / (cam.max() + 1e-3)  # 归一化到 [0, 1]
    
    return cam_img
    
def compute_weighted_CAM(activation, gamma=0.5, time=1.0):
    """
    Compute the weighted class activation map (CAM) based on the given activation tensor.

    Args:
        activation (torch.Tensor): The input activation tensor of shape (T, C, H, W),
                                   where T is the temporal dimension.
        gamma (float, optional): The decay rate for the weight computation. Default is 0.5.
        time (float, optional): A reference time for computing delta_t. Default is 1.0.

    Returns:
        torch.Tensor: A tensor representing the mean of all computed class activation maps (CAMs),
                      with shape (H, W).
    """
    # Ensure activation is a CUDA tensor
    # activation = activation.cuda()
    
    # Initialize variables
    previous_spike_time_list = activation.clone()
    weight = 0
    all_sam = []

    # Loop through the temporal dimension
    for prev_t in range(len(previous_spike_time_list)):
        delta_t = time - previous_spike_time_list[prev_t] * prev_t
        weight = weight +  torch.exp(gamma * (-1) * delta_t)

        weighted_activation = weight * activation
        sam = getForwardCAM(weighted_activation)  # Call the forward CAM function
        all_sam.append(sam)

    cam = torch.stack(all_sam).sum(dim=0)
    cam = cam - cam.min()  # 确保非负
    cam_img = cam / (cam.max() + 1e-3)
    # Stack all CAMs and return the mean
    return cam_img



def getForwardCAM(feature_conv):
    """
    计算前向类激活图 (Forward CAM) 的 PyTorch 实现。

    参数:
    - feature_conv (torch.Tensor): 卷积层特征图，形状为 (C, H, W)。

    返回:
    - List[torch.Tensor]: 归一化的类激活图，形状为 (H, W)。
    """
    # 对通道维度求和 (C -> 1)
    # sum(axis =0).sum(axis =0)
    cam = feature_conv.sum(dim=0).sum(axis=1)  # Shape: (H, W)
    
    # 激活值归一化处理
    cam = cam - cam.min()  # 确保非负
    cam_img = cam / (cam.max() + 1e-3)  # 归一化到 [0, 1]
    
    return cam_img

def getchannelCAM(feature_conv):
    """
    计算前向类激活图 (Forward CAM) 的 PyTorch 实现，仅对通道维度求和后进行归一化。

    参数:
    - feature_conv (torch.Tensor): 卷积层特征图，形状为 (C, H, W)。

    返回:
    - torch.Tensor: 归一化的类激活图，形状为 (H, W)。
    """
    # 对通道维度求和 (C -> 1)
    cam = feature_conv.sum(dim=0)  # Shape: (H, W)

    # 激活值归一化处理
    cam = cam - cam.min()  # 确保非负
    cam_img = cam / (cam.max() + 1e-3)  # 归一化到 [0, 1]

    return cam_img
    
import torch.nn.functional as F

def kl_divergence(Pa, Ps, temperature=1.0, reduction='batchmean'):
    """
    计算 Pa 和 Ps 之间的 KL 散度：KL(Pa || Ps)

    参数:
    - Pa (Tensor): 学生模型输出 logits，shape = [B, C]
    - Ps (Tensor): 教师模型输出 logits，shape = [B, C]
    - temperature (float): 温度系数，用于 softening 概率分布
    - reduction (str): 'batchmean' 或 'sum' 或 'none'

    返回:
    - loss (Tensor): KL 散度损失值
    """
    # 应用温度缩放并计算 Softmax 概率分布
    ps_log_prob = F.log_softmax(Ps / temperature, dim=1)
    pa_prob = F.softmax(Pa / temperature, dim=1)

    # 计算 KL 散度
    kl = F.kl_div(ps_log_prob, pa_prob, reduction=reduction)

    # 如果使用 temperature > 1，通常要乘上 temperature^2 来保持梯度尺度一致
    # 参考：https://huggingface.co/documentation/transformers/tasks/knowledge_distillation 
    if temperature > 1.0:
        kl *= temperature ** 2

    return kl

def soft_loss(ann_output, outputs, Tau=2.0):
    # 对教师模型的输出进行温度缩放后的softmax
    soft_targets = F.softmax(ann_output / Tau, dim=1)
    
    # 对学生模型的输出进行温度缩放
    soft_student_outputs = outputs / Tau
    
    # 计算交叉熵损失
    # 交叉熵的输入要求：学生模型的输出为logits，目标为概率分布
    ce_loss = F.cross_entropy(soft_student_outputs, soft_targets, reduction='mean')
    
    # 乘以 Tau^2 以平衡梯度
    scaled_ce_loss = ce_loss * (Tau ** 2)
    
    return scaled_ce_loss

def z_score_normalize(logit):
    mean = logit.mean(dim=-1, keepdims=True)
    stdv = logit.std(dim=-1, keepdims=True)
    return (logit - mean) / (1e-7 + stdv)
    
def soft_loss_feature(ann_output, outputs, T):
    
    # 对教师模型和学生模型的输出进行温度缩放后的softmax
    soft_targets = F.softmax(ann_output / T, dim=1)
    soft_student_outputs = F.log_softmax(outputs / T, dim=1)
    
    # 使用交叉熵计算软标签之间的损失
    loss = F.cross_entropy(soft_student_outputs, soft_targets) * (T * T)
    
    return loss


def visualize_cam(mask, img):
    """Make heatmap from mask and synthesize GradCAM result image using heatmap and img.
    Args:
        mask (torch.tensor): mask shape of (1, 1, H, W) and each element has value in range [0, 1]
        img (torch.tensor): img shape of (1, 3, H, W) and each pixel value is in range [0, 1]
        
    Return:
        heatmap (torch.tensor): heatmap img shape of (3, H, W)
        result (torch.tensor): synthesized GradCAM result of same shape with heatmap.
    """
    # Ensure mask is squeezed to (H, W) and converted to uint8
    mask_np = mask.squeeze().cpu().numpy()
    mask_np = np.uint8(255 * mask_np)  # Convert to uint8

    # Apply the heatmap (colormap) transformation
    heatmap = cv2.applyColorMap(mask_np, cv2.COLORMAP_JET)
    heatmap = torch.from_numpy(heatmap).permute(2, 0, 1).float().div(255)  # Convert to torch tensor and normalize
    b, g, r = heatmap.split(1)  # OpenCV uses BGR instead of RGB
    heatmap = torch.cat([r, g, b])  # Convert from BGR to RGB

    # Combine heatmap with the original image
    result = heatmap + img.cpu()
    result = result.div(result.max()).squeeze()  # Normalize the result
    
    return heatmap, result



def find_defulat_layer(arch, target_layer_name):
    target_layer = arch._modules[target_layer_name]
    return target_layer

def find_resnet_layer(arch, target_layer_name):
    """Find resnet layer to calculate GradCAM and GradCAM++
    
    Args:
        arch: default torchvision densenet models
        target_layer_name (str): the name of layer with its hierarchical information. please refer to usages below.
            target_layer_name = 'conv1'
            target_layer_name = 'layer1'
            target_layer_name = 'layer1_basicblock0'
            target_layer_name = 'layer1_basicblock0_relu'
            target_layer_name = 'layer1_bottleneck0'
            target_layer_name = 'layer1_bottleneck0_conv1'
            target_layer_name = 'layer1_bottleneck0_downsample'
            target_layer_name = 'layer1_bottleneck0_downsample_0'
            target_layer_name = 'avgpool'
            target_layer_name = 'fc'
            
    Return:
        target_layer: found layer. this layer will be hooked to get forward/backward pass information.
    """
    if 'layer' in target_layer_name:
        hierarchy = target_layer_name.split('_')
        layer_num = int(hierarchy[0].lstrip('layer'))
        if layer_num == 1:
            target_layer = arch.layer1
        elif layer_num == 2:
            target_layer = arch.layer2
        elif layer_num == 3:
            target_layer = arch.layer3
        elif layer_num == 4:
            target_layer = arch.layer4
        else:
            raise ValueError('unknown layer : {}'.format(target_layer_name))

        if len(hierarchy) >= 2:
            bottleneck_num = int(hierarchy[1].lower().lstrip('bottleneck').lstrip('basicblock'))
            target_layer = target_layer[bottleneck_num]

        if len(hierarchy) >= 3:
            target_layer = target_layer._modules[hierarchy[2]]
                
        if len(hierarchy) == 4:
            target_layer = target_layer._modules[hierarchy[3]]

    else:
        target_layer = arch._modules[target_layer_name]

    return target_layer


def find_densenet_layer(arch, target_layer_name):
    """Find densenet layer to calculate GradCAM and GradCAM++
    
    Args:
        arch: default torchvision densenet models
        target_layer_name (str): the name of layer with its hierarchical information. please refer to usages below.
            target_layer_name = 'features'
            target_layer_name = 'features_transition1'
            target_layer_name = 'features_transition1_norm'
            target_layer_name = 'features_denseblock2_denselayer12'
            target_layer_name = 'features_denseblock2_denselayer12_norm1'
            target_layer_name = 'features_denseblock2_denselayer12_norm1'
            target_layer_name = 'classifier'
            
    Return:
        target_layer: found layer. this layer will be hooked to get forward/backward pass information.
    """
    
    hierarchy = target_layer_name.split('_')
    target_layer = arch._modules[hierarchy[0]]

    if len(hierarchy) >= 2:
        target_layer = target_layer._modules[hierarchy[1]]

    if len(hierarchy) >= 3:
        target_layer = target_layer._modules[hierarchy[2]]

    if len(hierarchy) == 4:
        target_layer = target_layer._modules[hierarchy[3]]

    return target_layer


def find_vgg_layer(arch, target_layer_name):
    """Find vgg layer to calculate GradCAM and GradCAM++
    
    Args:
        arch: default torchvision densenet models
        target_layer_name (str): the name of layer with its hierarchical information. please refer to usages below.
            target_layer_name = 'features'
            target_layer_name = 'features_42'
            target_layer_name = 'classifier'
            target_layer_name = 'classifier_0'
            
    Return:
        target_layer: found layer. this layer will be hooked to get forward/backward pass information.
    """
    hierarchy = target_layer_name.split('_')

    if len(hierarchy) >= 1:
        target_layer = arch.features

    if len(hierarchy) == 2:
        target_layer = target_layer[int(hierarchy[1])]

    return target_layer


def find_alexnet_layer(arch, target_layer_name):
    """Find alexnet layer to calculate GradCAM and GradCAM++
    
    Args:
        arch: default torchvision densenet models
        target_layer_name (str): the name of layer with its hierarchical information. please refer to usages below.
            target_layer_name = 'features'
            target_layer_name = 'features_0'
            target_layer_name = 'classifier'
            target_layer_name = 'classifier_0'
            
    Return:
        target_layer: found layer. this layer will be hooked to get forward/backward pass information.
    """
    hierarchy = target_layer_name.split('_')

    if len(hierarchy) >= 1:
        target_layer = arch.features

    if len(hierarchy) == 2:
        target_layer = target_layer[int(hierarchy[1])]

    return target_layer


def find_squeezenet_layer(arch, target_layer_name):
    """Find squeezenet layer to calculate GradCAM and GradCAM++
    
    Args:
        arch: default torchvision densenet models
        target_layer_name (str): the name of layer with its hierarchical information. please refer to usages below.
            target_layer_name = 'features_12'
            target_layer_name = 'features_12_expand3x3'
            target_layer_name = 'features_12_expand3x3_activation'
            
    Return:
        target_layer: found layer. this layer will be hooked to get forward/backward pass information.
    """
    hierarchy = target_layer_name.split('_')
    target_layer = arch._modules[hierarchy[0]]

    if len(hierarchy) >= 2:
        target_layer = target_layer._modules[hierarchy[1]]

    if len(hierarchy) == 3:
        target_layer = target_layer._modules[hierarchy[2]]

    elif len(hierarchy) == 4:
        target_layer = target_layer._modules[hierarchy[2]+'_'+hierarchy[3]]

    return target_layer


def denormalize(tensor, mean, std):
    if not tensor.ndimension() == 4:
        raise TypeError('tensor should be 4D')

    mean = torch.FloatTensor(mean).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)
    std = torch.FloatTensor(std).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)

    return tensor.mul(std).add(mean)


def normalize(tensor, mean, std):
    if not tensor.ndimension() == 4:
        raise TypeError('tensor should be 4D')

    mean = torch.FloatTensor(mean).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)
    std = torch.FloatTensor(std).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)

    return tensor.sub(mean).div(std)


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return self.do(tensor)
    
    def do(self, tensor):
        return normalize(tensor, self.mean, self.std)
    
    def undo(self, tensor):
        return denormalize(tensor, self.mean, self.std)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
    

class HookManager:
    def __init__(self, model, target_layer_name):
        """
        初始化 HookManager
        :param model: PyTorch 模型 (例如 ResNet-18)
        :param target_layer_name: 要注册 hook 的目标层名称 (例如 'layer4')
        """
        self.model = model
        self.target_layer_name = target_layer_name
        self.hook = None
        self.outputs = None  # 用于存储目标层的输出

    def register_hook(self):
        """
        注册 hook 到目标层，提取其输出
        """
        if not hasattr(self.model, self.target_layer_name):
            raise ValueError(f"Layer {self.target_layer_name} not found in the model.")
        
        target_layer = getattr(self.model, self.target_layer_name)
        
        def hook_fn(module, input, output):
            self.outputs = output  # 保存目标层的输出

        # 注册 hook
        self.hook = target_layer.register_forward_hook(hook_fn)
        # print(f"Hook registered on layer: {self.target_layer_name}")

    def remove_hook(self):
        """
        移除已注册的 hook
        """
        if self.hook is not None:
            self.hook.remove()
            self.hook = None
            # print(f"Hook removed from layer: {self.target_layer_name}")
        # else:
            # print("No hook to remove.")

    def get_outputs(self):
        """
        获取目标层的输出
        :return: 目标层的输出张量
        """
        if self.outputs is None:
            print("No output available. Ensure the forward pass is run after registering the hook.")
        return self.outputs
    
def count_parameters(model):
    """统计模型参数总量
    
    Args:
        model: PyTorch模型
        
    Returns:
        total_params: 参数总量
        trainable_params: 可训练参数量
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # 转换为M为单位
    total_params_M = total_params / 1e6
    trainable_params_M = trainable_params / 1e6
    
    print(f'Total Parameters: {total_params_M:.2f}M')
    print(f'Trainable Parameters: {trainable_params_M:.2f}M')
    