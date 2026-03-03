import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# =============================================================================
# 9.1 跨模態對齊 (Cross-modal Alignment)
# 模擬 CLIP 結構：影像編碼器與文本編碼器將資料映射至共同向量空間 (Joint Vector Space)
# =============================================================================

class CrossModalModel(nn.Module):
    def __init__(self, visual_dim, textual_dim, joint_dim):
        super(CrossModalModel, self).__init__()
        # 影像編碼器 f_v
        self.visual_encoder = nn.Linear(visual_dim, joint_dim)
        # 文本編碼器 f_w
        self.textual_encoder = nn.Linear(textual_dim, joint_dim)
        # 溫度參數 tau
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, image_features, text_features):
        # 得到影像與文本的嵌入向量 v_i, w_i
        v = self.visual_encoder(image_features)
        w = self.textual_encoder(text_features)
        
        # 正規化以計算餘弦相似度 (Cosine Similarity)
        v = v / v.norm(dim=-1, keepdim=True)
        w = w / w.norm(dim=-1, keepdim=True)
        
        return v, w

def contrastive_loss(v, w, logit_scale):
    """
    實作 InfoNCE Loss
    """
    tau = torch.exp(logit_scale)
    # 計算相似度矩陣 s_{i,j} = v_i^T w_j
    logits = torch.matmul(v, w.t()) * tau
    labels = torch.arange(len(v)).to(v.device)
    
    # 對影像與文本雙向計算交叉熵
    loss_v = F.cross_entropy(logits, labels)
    loss_w = F.cross_entropy(logits.t(), labels)
    return (loss_v + loss_w) / 2

# =============================================================================
# 9.2 潛在擴散模型 (Latent Diffusion Models)
# 模擬條件生成中的雜訊預測器 eps_theta(z_t, t, c)
# =============================================================================

class SimpleUNet(nn.Module):
    def __init__(self, latent_dim, cond_dim):
        super(SimpleUNet, self).__init__()
        # 模擬 U-Net 的簡化結構，將潛在變數與條件(文本)結合
        self.net = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )

    def forward(self, z_t, t, c):
        # 將時間步長 t 擴展後與 z_t 及條件 c 拼接
        # 在此簡化直接將 z_t 與 c 拼接
        x = torch.cat([z_t, c], dim=-1)
        return self.net(x)

# =============================================================================
# 9.3 檢索增強生成 (Retrieval-Augmented Generation, RAG)
# 實作檢索函數 q(y | x, D) 的向量檢索部分
# =============================================================================

class VectorDatabase:
    def __init__(self, dimension):
        self.dimension = dimension
        self.vectors = []
        self.documents = []

    def add_document(self, doc, embedding_model):
        self.documents.append(doc)
        # 模擬 Embedding Model E(d)
        embedding = embedding_model(doc)
        self.vectors.append(embedding)

    def query(self, query_vector, k=1):
        # 使用餘弦相似度進行檢索 sim(E(q), E(d))
        db_tensor = torch.stack(self.vectors)
        # 相似度計算
        similarities = F.cosine_similarity(query_vector, db_tensor)
        top_k_indices = torch.topk(similarities, k).indices
        return [self.documents[i] for i in top_k_indices]

# =============================================================================
# 執行範例
# =============================================================================

if __name__ == "__main__":
    # --- 測試 9.1 對比學習 ---
    print("--- 9.1 跨模態對齊範例 ---")
    batch_size, v_dim, t_dim, j_dim = 4, 512, 384, 256
    model = CrossModalModel(v_dim, t_dim, j_dim)
    
    img_data = torch.randn(batch_size, v_dim)
    txt_data = torch.randn(batch_size, t_dim)
    
    v_embed, w_embed = model(img_data, txt_data)
    loss = contrastive_loss(v_embed, w_embed, model.logit_scale)
    print(f"對比損失值 (Contrastive Loss): {loss.item():.4f}\n")

    # --- 測試 9.2 擴散模型 ---
    print("--- 9.2 潛在擴散模型雜訊預測 ---")
    latent_dim = 64
    cond_dim = j_dim # 使用剛剛對比學習得到的文本維度
    diffusion_net = SimpleUNet(latent_dim, cond_dim)
    
    z_t = torch.randn(1, latent_dim) # 潛在空間的雜訊影像
    c = w_embed[0:1]                 # 文本條件向量
    t = torch.tensor([0.5])          # 時間步長
    
    epsilon_theta = diffusion_net(z_t, t, c)
    print(f"預測的雜訊向量 (Epsilon Theta) 前5維: {epsilon_theta[0, :5].detach().numpy()}\n")

    # --- 測試 9.3 RAG 檢索 ---
    print("--- 9.3 檢索增強生成 (RAG) 模擬 ---")
    # 模擬簡單的 Embedding 函數
    def simple_embedding(text):
        # 隨機生成向量來代表語義特徵
        torch.manual_seed(len(text))
        return torch.randn(j_dim)

    db = VectorDatabase(dimension=j_dim)
    db.add_document("微積分是研究極限的數學。", simple_embedding)
    db.add_document("線性代數處理向量空間與線性映射。", simple_embedding)
    
    user_query = "什麼是線性代數？"
    query_vec = simple_embedding(user_query).unsqueeze(0)
    retrieved_docs = db.query(query_vec, k=1)
    
    print(f"使用者查詢: {user_query}")
    print(f"檢索到的文檔: {retrieved_docs[0]}")