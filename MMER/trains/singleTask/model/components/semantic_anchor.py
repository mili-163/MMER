import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple

class DualLevelSemanticAnchoring:
    """
    Dual-Level Semantic Anchoring
    - local anchors: 每个类别每个模态的top-K高置信样本均值
    - global anchors: 每个类别结构增强特征的top-K均值
    - category prototypes: LLM原型
    """
    def __init__(self, top_k: int = 15, lambda_entropy: float = 0.1):
        self.top_k = top_k
        self.lambda_entropy = lambda_entropy

    @staticmethod
    def entropy(probs: torch.Tensor, eps=1e-8):
        # probs: [N, C]
        return -torch.sum(probs * torch.log(probs + eps), dim=-1)  # [N]

    def compute_similarity(self, z: torch.Tensor, p: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        # z: [N, D], p: [C, D], alpha: [N, C]
        # 返回 [N, C]，每个样本与每个类别原型的相似度
        z_norm = F.normalize(z, dim=-1)  # [N, D]
        p_norm = F.normalize(p, dim=-1)  # [C, D]
        sim = torch.matmul(z_norm, p_norm.t())  # [N, C]
        ent = self.entropy(alpha)  # [N]
        sim = sim + self.lambda_entropy * ent.unsqueeze(1)  # broadcast
        return sim

    def get_topk_indices(self, sim: torch.Tensor, k: int) -> List[List[int]]:
        # sim: [N, C]
        # 返回每个类别的top-K样本索引
        topk_idx = []
        for c in range(sim.shape[1]):
            idx = torch.topk(sim[:, c], k=min(k, sim.shape[0]), largest=True).indices.tolist()
            topk_idx.append(idx)
        return topk_idx  # List[C][K]

    def compute_local_anchors(self, z: torch.Tensor, p: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        # z: [N, D], p: [C, D], alpha: [N, C]
        sim = self.compute_similarity(z, p, alpha)  # [N, C]
        topk_idx = self.get_topk_indices(sim, self.top_k)  # List[C][K]
        anchors = []
        for c, idxs in enumerate(topk_idx):
            if len(idxs) == 0:
                anchors.append(torch.zeros(z.shape[1], device=z.device))
            else:
                anchors.append(z[idxs].mean(dim=0))
        return torch.stack(anchors, dim=0)  # [C, D]

    def compute_global_anchors(self, u: torch.Tensor, p: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        # u: [N, D], p: [C, D], alpha: [N, C]
        sim = self.compute_similarity(u, p, alpha)  # [N, C]
        topk_idx = self.get_topk_indices(sim, self.top_k)  # List[C][K]
        anchors = []
        for c, idxs in enumerate(topk_idx):
            if len(idxs) == 0:
                anchors.append(torch.zeros(u.shape[1], device=u.device))
            else:
                anchors.append(u[idxs].mean(dim=0))
        return torch.stack(anchors, dim=0)  # [C, D]

    def get_all_anchors(self, modal_features: Dict[str, torch.Tensor],
                        fused_features: torch.Tensor,
                        category_prototypes: torch.Tensor,
                        pseudo_labels: Dict[str, torch.Tensor],
                        fused_pseudo_label: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        计算所有local/global anchors
        返回：{
            'local_anchors': {mod: [C, D]},
            'global_anchors': [C, D],
            'category_prototypes': [C, D]
        }
        """
        anchors = {'local_anchors': {}, 'global_anchors': None, 'category_prototypes': category_prototypes}
        for mod in modal_features:
            if modal_features[mod] is not None and pseudo_labels[mod] is not None:
                anchors['local_anchors'][mod] = self.compute_local_anchors(
                    modal_features[mod], category_prototypes, pseudo_labels[mod])
        anchors['global_anchors'] = self.compute_global_anchors(
            fused_features, category_prototypes, fused_pseudo_label)
        return anchors 