import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

class UtilityRewardModel:
    """语义效用模型，使用Sentence-BERT计算Embedding相似度"""
    def __init__(self, model_path="BAAI/bge-m3", device="cuda"):
        print(f"[Utility] Loading Utility Model from {model_path}...")
        self.device = device
        self.model = SentenceTransformer(model_path, device=device)
        self.model.eval()

    def compute_score(self, original_texts, anonymized_texts):
        """
        计算语义相似度Reward
        
        Args:
            original_texts: 原始文本列表
            anonymized_texts: 脱敏后文本列表
        Returns:
            torch.Tensor: 相似度分数 (batch_size,), 范围[0, 1]
        """
        # 1. 批量Encode
        with torch.no_grad():
            embeddings_orig = self.model.encode(original_texts, convert_to_tensor=True)
            embeddings_anon = self.model.encode(anonymized_texts, convert_to_tensor=True)

        # 2. 计算余弦相似度
        scores = F.cosine_similarity(embeddings_orig, embeddings_anon, dim=1)

        # 3. 非线性缩放 (相似度0.9->0.72, 0.6->0.21)
        scaled_scores = torch.pow(scores, 3)
        
        return scaled_scores.to(self.device)