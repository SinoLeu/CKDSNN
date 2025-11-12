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
    
    return kl_loss


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



    