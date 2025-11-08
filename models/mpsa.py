import math
import torch
import torch.nn as nn
from einops import rearrange
from os import makedirs
import torch
import torch.nn as nn
import torch.nn.functional as F


# class AttentionFeatureFusion(nn.Module):
#     def __init__(self, input_dims, embedding_dim):
#         """
#         初始化注意力特征融合模块。
#         :param input_dims: 每个特征的输入维度列表，例如 [256, 512, 1024]
#         :param embedding_dim: 融合后特征的目标维度
#         """
#         super(AttentionFeatureFusion, self).__init__()
#         self.input_dims = input_dims
        
#         # 定义特征变换层（将每个特征映射到统一维度）
#         self.feature_transforms = nn.ModuleList([
#             nn.Linear(in_features=dim, out_features=embedding_dim)
#             for dim in input_dims
#         ])
        
#         # 定义注意力评分网络
#         self.attention = nn.Sequential(
#             nn.Linear(embedding_dim, 1),  # 将特征映射到标量
#             nn.Softmax(dim=0)  # 对所有特征的注意力分数归一化
#         )

#     def forward(self, features):
#         """
#         前向传播：基于注意力机制的特征融合。
#         :param features: 输入特征列表，形状为 [[B, D1], [B, D2], ..., [B, DN]]
#         :return: 融合后的特征，形状为 [B, embedding_dim]
#         """
#         assert len(features) == len(self.input_dims), "特征数量与初始化时不一致！"

#         # (1) 对每个特征进行变换，统一维度
#         transformed_features = []
#         for i, feat in enumerate(features):
#             transformed_feat = self.feature_transforms[i](feat)  # [B, embedding_dim]
#             transformed_features.append(transformed_feat)

#         # (2) 计算注意力权重
#         attention_scores = []
#         for feat in transformed_features:
#             score = self.attention(feat)  # [B, 1]
#             attention_scores.append(score)

#         attention_scores = torch.stack(attention_scores, dim=0)  # [N, B, 1]

#         # (3) 加权融合
#         fused_feature = torch.zeros_like(transformed_features[0])  # 初始化融合特征 [B, embedding_dim]
#         for i, feat in enumerate(transformed_features):
#             fused_feature += feat * attention_scores[i]  # 加权求和

#         return fused_feature
    
class PartAttention(nn.Module):
    """部件注意力模块"""
    def __init__(self, in_channels, num_parts=8):
        super().__init__()
        self.num_parts = num_parts
        self.part_scores = nn.Conv2d(in_channels, num_parts, kernel_size=1)
        
    def forward(self, x):
        # 生成部件注意力图
        scores = self.part_scores(x)  # B×P×H×W
        part_maps = F.softmax(scores.view(*scores.size()[:2], -1), dim=2)
        part_maps = part_maps.view_as(scores)
        
        # 保存部件注意力图用于损失计算
        self.attention_maps = part_maps
        
        # 提取部件特征
        part_features = []
        for i in range(self.num_parts):
            masked_feature = x * part_maps[:, i:i+1, :, :]
            pooled_feature = F.adaptive_avg_pool2d(masked_feature, 1)
            part_features.append(pooled_feature.squeeze(-1).squeeze(-1))
            
        # 连接所有部件特征
        return torch.cat(part_features, dim=1)
    
    def get_diversity_loss(self):
        """计算部件多样性损失"""
        if not hasattr(self, 'attention_maps'):
            return torch.tensor(0.0).to(next(self.parameters()).device)
            
        batch_size, num_parts = self.attention_maps.size(0), self.attention_maps.size(1)
        part_maps_flat = self.attention_maps.view(batch_size, num_parts, -1)
        
        # 计算部件之间的相似度
        normalized_maps = F.normalize(part_maps_flat, p=2, dim=2)
        similarity = torch.matmul(normalized_maps, normalized_maps.transpose(1, 2))
        
        # 排除自相似度
        mask = torch.ones_like(similarity) - torch.eye(num_parts).unsqueeze(0).to(similarity.device)
        diversity_loss = (similarity * mask).sum() / (num_parts * (num_parts - 1))
        
        return diversity_loss

class MultiStageFusion(nn.Module):
    def __init__(self, channel_dim_list, dropouts=[0.1, 0.1, 0.1, 0.1], output_size=(1, 1), fusion_dim=768, 
                 use_part_attention=True, num_parts=8):
        super().__init__()
        
        
        # 记录是否使用部件注意力
        self.use_part_attention = use_part_attention
        self.num_parts = num_parts
        
        # 卷积层：调整通道数
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            for in_channels, out_channels in zip(channel_dim_list, channel_dim_list)
        ])
        
        # 归一化层
        self.norm_layers = nn.ModuleList([
            nn.BatchNorm2d(out_channels) for out_channels in channel_dim_list
        ])
        
        # Dropout 层
        self.dropout_layers = nn.ModuleList([
            nn.Dropout(p=dropout) for dropout in dropouts[:len(channel_dim_list)]
        ])
        
        # 自适应池化层：统一空间尺寸
        self.pool = nn.AdaptiveAvgPool2d(output_size)
        
        # 添加部件注意力模块
        if use_part_attention:
            self.part_attention_modules = nn.ModuleList([
                PartAttention(in_channels=dim, num_parts=num_parts) 
                for dim in channel_dim_list
            ])
            
            # 融合后的维度：原始特征 + 部件特征
            self.fused_dim = sum(channel_dim_list) + sum(channel_dim_list) * num_parts
            input_dims = channel_dim_list + [dim * num_parts for dim in channel_dim_list]
            # self.att_fusion = AttentionFeatureFusion(input_dims=input_dims,embedding_dim=self.fused_dim)
        else:
            self.fused_dim = sum(channel_dim_list)
            # input_dims = channel_dim_list
            # self.att_fusion = AttentionFeatureFusion(input_dims=channel_dim_list,embedding_dim=self.fused_dim)
        # 融合层
        self.fusion_layer = nn.Linear(self.fused_dim, fusion_dim)

    def forward(self, features):
        # print(len(features))
        processed_features = []
        part_features = []
        
        for i, (stage_feat, conv, norm, dropout) in enumerate(zip(features, self.conv_layers, self.norm_layers, self.dropout_layers)):
            # 常规特征处理
            x = conv(stage_feat)  # [batch, channel_dim_list[i], h_i, w_i]
            x = norm(x)           # 归一化
            x = F.relu(x, inplace=True)  # 激活
            x_pool = self.pool(x)      # 自适应池化到 [batch, channel_dim_list[i], 1, 1]
            x_pool = dropout(x_pool)        # Dropout
            x_pool = x_pool.view(x_pool.size(0), -1)  # 展平为 [batch, channel_dim_list[i]]
            processed_features.append(x_pool)
            
            # 部件注意力处理
            if self.use_part_attention:
                part_feat = self.part_attention_modules[i](x)  # [batch, channel_dim_list[i] * num_parts]
                part_features.append(part_feat)
        
        # 特征融合
        if self.use_part_attention:
            # print(processed_features[0].shape,processed_features[1].shape,processed_features[2].shape)
            all_features = processed_features + part_features
            fused_feature = torch.cat(all_features, dim=1)  # [batch, fused_dim]
            # print(part_features[0].shape)
            # fused_feature = self.att_fusion(all_features)
            # print(fused_feature.shape)
            
        else:
            fused_feature = torch.cat(processed_features, dim=1)
            # fused_feature = self.att_fusion(processed_features)
        output = self.fusion_layer(fused_feature)  # [batch, fusion_dim]
        
        return output
    
    def get_diversity_loss(self):
        """计算所有部件注意力模块的多样性损失"""
        if not self.use_part_attention:
            return torch.tensor(0.0).to(next(self.parameters()).device)
            
        total_loss = 0.0
        for module in self.part_attention_modules:
            total_loss += module.get_diversity_loss()
        
        return total_loss / len(self.part_attention_modules)

class MultiStageFeatureModule(nn.Module):
    def __init__(self, nb_class, channel_dim_list, dropouts=[0.1, 0.1, 0.1, 0.1], output_size=(1, 1), fusion_dim=768,
                 use_part_attention=True, num_parts=8):
        super().__init__()
        self.multi_stage_fusion = MultiStageFusion(
            channel_dim_list=channel_dim_list,
            dropouts=dropouts,
            output_size=output_size,
            fusion_dim=fusion_dim,
            use_part_attention=use_part_attention,
            num_parts=num_parts
        )
        self.head = nn.Linear(fusion_dim, nb_class)
        
    def forward(self, x):
        out = self.multi_stage_fusion(x)
        return self.head(out)
    
    def get_diversity_loss(self):
        """获取部件多样性损失，用于训练"""
        return self.multi_stage_fusion.get_diversity_loss()
    
# if __name__ == '__main__':
#     ## QKFormer_10_384 torch.Size([2, 96, 56, 56]) torch.Size([2, 192, 28, 28]) torch.Size([2, 192, 14, 14])
#     ## QKFormer_10_512 torch.Size([2, 128, 56, 56]) torch.Size([2, 256, 28, 28]) torch.Size([2, 512, 14, 14])
#     ## QKFormer_10_512 torch.Size([2, 192, 56, 56]) torch.Size([2, 384, 28, 28]) torch.Size([2, 768, 14, 14])
#     #    channel_dim_list = [192, 384, 768] 
#    channel_dim_list = [96, 192, 192] 
#    da_module = MultiStageFeatureModule(nb_class=200,channel_dim_list=channel_dim_list)
   
#    t1 = torch.rand(3,96,56,56)
#    t2 = torch.rand(3,192,28,28)
#    t3 = torch.rand(3,192,14,14)
#    out = da_module([t1,t2,t3])
#    print(out.shape)