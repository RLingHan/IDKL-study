import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1. / p)


class GraphContrastiveAlignment(nn.Module):
    """
    GCA: 基于身份的模态不变拓扑对齐

    核心理解：
    - V和I样本不是一一对应的（不同视角）
    - 但同ID的V样本集和I样本集应该有相同的"群体结构"
    - 比较的是：ID级别的拓扑关系，而非样本级别
    """

    def __init__(self, k_neighbors=10, temperature=0.15):
        super(GraphContrastiveAlignment, self).__init__()
        self.k = k_neighbors
        self.temperature = temperature

    def identity_level_topology_alignment(self, feat_v, feat_i, labels_v, labels_i):
        """
        策略1：身份级别拓扑对齐（核心创新）

        思路：
        1. 不比较单个V样本和单个I样本（因为不对应）
        2. 比较ID_A的V样本集 和 ID_A的I样本集 的内部结构
        3. 再比较ID_A和ID_B之间的关系在V和I中是否一致
        """
        unique_ids = torch.unique(labels_v)
        if len(unique_ids) < 2:
            return torch.tensor(0.0, device=feat_v.device)

        # 为每个ID计算其在V和I模态下的"中心表示"和"分布特性"
        id_centers_v = []
        id_centers_i = []
        id_spreads_v = []  # 类内散度
        id_spreads_i = []
        valid_ids = []

        for pid in unique_ids:
            mask_v = (labels_v == pid)
            mask_i = (labels_i == pid)

            if mask_v.sum() < 2 or mask_i.sum() < 2:
                continue

            fv = feat_v[mask_v]  # [N_v, D]
            fi = feat_i[mask_i]  # [N_i, D]

            # 计算中心
            center_v = fv.mean(dim=0)  # [D]
            center_i = fi.mean(dim=0)  # [D]

            # 计算类内散度（标准差）
            spread_v = torch.std(fv, dim=0).mean()  # scalar
            spread_i = torch.std(fi, dim=0).mean()  # scalar

            id_centers_v.append(center_v)
            id_centers_i.append(center_i)
            id_spreads_v.append(spread_v)
            id_spreads_i.append(spread_i)
            valid_ids.append(pid)

        if len(valid_ids) < 2:
            return torch.tensor(0.0, device=feat_v.device)

        # 转换为tensor
        centers_v = torch.stack(id_centers_v)  # [num_ids, D]
        centers_i = torch.stack(id_centers_i)  # [num_ids, D]
        spreads_v = torch.stack(id_spreads_v)  # [num_ids]
        spreads_i = torch.stack(id_spreads_i)  # [num_ids]

        # Loss 1: ID中心之间的拓扑对齐
        # 计算ID间的相似度矩阵
        centers_v_norm = F.normalize(centers_v, p=2, dim=1)
        centers_i_norm = F.normalize(centers_i, p=2, dim=1)

        sim_matrix_v = torch.mm(centers_v_norm, centers_v_norm.t())  # [num_ids, num_ids]
        sim_matrix_i = torch.mm(centers_i_norm, centers_i_norm.t())  # [num_ids, num_ids]

        # 对齐ID间的关系结构
        loss_inter_id = F.mse_loss(sim_matrix_v, sim_matrix_i)

        # Loss 2: 类内散度对齐（创新点）
        # 同一个ID在V和I中的"紧密程度"应该一致
        loss_intra_spread = F.mse_loss(spreads_v, spreads_i)

        return loss_inter_id + 0.5 * loss_intra_spread

    def intra_identity_consistency(self, feat_v, feat_i, labels_v, labels_i):
        """
        策略2：类内一致性约束（保留你的有效发现）

        思路：
        - 同ID的V样本之间的相似性分布
        - 应该和同ID的I样本之间的相似性分布一致
        - 这是你说的"误打误撞"的本质
        """
        unique_ids = torch.unique(labels_v)
        total_loss = 0.0
        valid_count = 0

        for pid in unique_ids:
            mask_v = (labels_v == pid)
            mask_i = (labels_i == pid)

            fv = feat_v[mask_v]
            fi = feat_i[mask_i]

            if fv.size(0) < 2 or fi.size(0) < 2:
                continue

            # 归一化
            fv_norm = F.normalize(fv, p=2, dim=1)
            fi_norm = F.normalize(fi, p=2, dim=1)

            # 类内相似度矩阵
            sim_v = torch.mm(fv_norm, fv_norm.t())  # [N_v, N_v]
            sim_i = torch.mm(fi_norm, fi_norm.t())  # [N_i, N_i]

            # 匹配尺寸
            min_size = min(sim_v.size(0), sim_i.size(0))
            sim_v = sim_v[:min_size, :min_size]
            sim_i = sim_i[:min_size, :min_size]

            # 相似度分布对齐
            loss = F.mse_loss(sim_v, sim_i)
            total_loss += loss
            valid_count += 1

        if valid_count == 0:
            return torch.tensor(0.0, device=feat_v.device)

        return total_loss / valid_count

    def cross_identity_ranking_consistency(self, feat_v, feat_i, labels_v, labels_i):
        """
        策略3：跨身份排序一致性（核心创新）

        思路：
        对于任意ID_A的某个V样本：
        1. 找到其在V模态中最相似的K个其他ID（可能包含ID_B, ID_C...）
        2. 对ID_A的任意I样本，也找其最相似的K个其他ID
        3. 这两个"ID邻居集合"应该有较大重叠

        关键：比较的是ID级别的邻域，而非样本级别
        """
        unique_ids = torch.unique(labels_v)
        if len(unique_ids) < 3:  # 至少需要3个ID才能比较邻域
            return torch.tensor(0.0, device=feat_v.device)

        total_loss = 0.0
        valid_count = 0

        # 为每个ID计算其"ID级别的邻居"
        for pid in unique_ids:
            mask_v = (labels_v == pid)
            mask_i = (labels_i == pid)

            if mask_v.sum() == 0 or mask_i.sum() == 0:
                continue

            fv = feat_v[mask_v]  # 该ID的所有V样本
            fi = feat_i[mask_i]  # 该ID的所有I样本

            # 计算该ID与其他所有样本的平均相似度（在V和I中分别计算）
            fv_norm = F.normalize(fv, p=2, dim=1)
            fi_norm = F.normalize(fi, p=2, dim=1)

            # V模态：该ID的每个样本到所有其他ID样本的平均相似度
            other_mask_v = (labels_v != pid)
            if other_mask_v.sum() == 0:
                continue

            other_fv = feat_v[other_mask_v]
            other_labels_v = labels_v[other_mask_v]
            other_fv_norm = F.normalize(other_fv, p=2, dim=1)

            # 计算该ID的V样本集与其他每个ID的平均相似度
            sim_v = torch.mm(fv_norm, other_fv_norm.t())  # [N_v, N_other]
            id_level_sim_v = []
            for other_pid in torch.unique(other_labels_v):
                mask_other = (other_labels_v == other_pid)
                avg_sim = sim_v[:, mask_other].mean()
                id_level_sim_v.append(avg_sim)
            id_level_sim_v = torch.stack(id_level_sim_v)  # [num_other_ids]

            # I模态：同样的计算
            other_mask_i = (labels_i != pid)
            if other_mask_i.sum() == 0:
                continue

            other_fi = feat_i[other_mask_i]
            other_labels_i = labels_i[other_mask_i]
            other_fi_norm = F.normalize(other_fi, p=2, dim=1)

            sim_i = torch.mm(fi_norm, other_fi_norm.t())  # [N_i, N_other]
            id_level_sim_i = []
            for other_pid in torch.unique(other_labels_i):
                mask_other = (other_labels_i == other_pid)
                avg_sim = sim_i[:, mask_other].mean()
                id_level_sim_i.append(avg_sim)
            id_level_sim_i = torch.stack(id_level_sim_i)  # [num_other_ids]

            # 对齐：该ID在V和I中的"ID级邻域"应该一致
            min_len = min(len(id_level_sim_v), len(id_level_sim_i))
            if min_len > 0:
                loss = F.mse_loss(
                    id_level_sim_v[:min_len],
                    id_level_sim_i[:min_len]
                )
                total_loss += loss
                valid_count += 1

        if valid_count == 0:
            return torch.tensor(0.0, device=feat_v.device)

        return total_loss / valid_count

    def forward(self, feat_v, feat_i, labels, is_shared=True):
        """
        前向传播

        注意：labels是V模态的标签，但V和I应该有相同的ID分布
        （因为是交替采样的）
        """
        if feat_v.size(0) == 0 or feat_i.size(0) == 0:
            return torch.tensor(0.0, device=feat_v.device, requires_grad=True)

        # 展平特征
        if feat_v.dim() > 2:
            feat_v = gem(feat_v).squeeze()
            feat_v = feat_v.view(feat_v.size(0), -1)
        if feat_i.dim() > 2:
            feat_i = gem(feat_i).squeeze()
            feat_i = feat_i.view(feat_i.size(0), -1)

        # 假设V和I的labels顺序一致（因为交替采样）
        labels_v = labels
        labels_i = labels

        total_loss = 0.0

        if is_shared:
            # 共享特征：注重ID级别的全局对齐
            loss1 = self.identity_level_topology_alignment(feat_v, feat_i, labels_v, labels_i)
            loss2 = self.cross_identity_ranking_consistency(feat_v, feat_i, labels_v, labels_i)
            total_loss = loss1 + 0.5 * loss2

        else:
            # 特定特征：注重类内一致性（保留你的有效发现）
            loss3 = self.intra_identity_consistency(feat_v, feat_i, labels_v, labels_i)
            total_loss = loss3

        # 安全检查
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            total_loss = torch.tensor(0.0, device=feat_v.device, requires_grad=True)

        total_loss = torch.clamp(total_loss, min=0.0)

        return total_loss