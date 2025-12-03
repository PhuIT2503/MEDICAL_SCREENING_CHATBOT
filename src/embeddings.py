import torch
import numpy as np
import re
from typing import List, Any
from transformers import AutoModel, AutoTokenizer
from langchain_core.embeddings import Embeddings
from underthesea import word_tokenize
from . import config
import pandas as pd

# Chọn thiết bị chạy model: GPU nếu có, nếu không thì CPU
device = torch.device("gpu" if torch.cuda.is_available() else "cpu")

# Lưu biến tokenizer và model để tránh load lại nhiều lần
model_components = None

def load_model_components():
    """
    Tải model và tokenizer PhoBERT từ thư mục fine-tuned.
    Chỉ tải 1 lần duy nhất, các lần sau dùng lại để tối ưu hiệu năng.
    """
    global model_components
    if model_components is None:
        print("Đang tải tokenizer và model...")

        # Load tokenizer PhoBERT
        tokenizer = AutoTokenizer.from_pretrained(config.model_path)

        # Load model PhoBERT
        model = AutoModel.from_pretrained(config.model_path)

        model_components = {"tokenizer": tokenizer, "model": model}

    return model_components


def preprocess_text(text: Any) -> str:
    """
    Tiền xử lý văn bản:
    - Kiểm tra giá trị NaN hoặc rỗng
    - Lowercase
    - Loại bỏ ký tự thừa, xuống dòng
    - Tokenize bằng underthesea (tách từ kiểu tiếng Việt)
    """
    if pd.isna(text) or not str(text).strip():
        return ""

    text = str(text).lower().replace('\n', ' ').replace('\r', '')
    text = re.sub(' +', ' ', text).strip()

    if not text:
        return ""

    return word_tokenize(text, format="text")


def get_embedding(text: str) -> np.ndarray:
    """
    Tính embedding cho văn bản bằng PhoBERT:
    - Tokenize input
    - Truyền qua model
    - Mean pooling (dựa trên attention mask)
    - Trả về vector embedding dạng numpy float32
    """
    components = load_model_components()
    tokenizer = components["tokenizer"]
    model = components["model"]

    # Tiền xử lý và tách từ
    segmented_text = preprocess_text(text)

    # Trường hợp văn bản rỗng → trả về vector 0
    if not segmented_text:
        return np.zeros(config.embedding_dim).astype("float32")

    # Tokenize input
    inputs = tokenizer(
        segmented_text,
        return_tensors="pt",
        truncation=True,
        max_length=config.max_length,
        padding="max_length"
    )

    # Đưa tensor sang GPU (nếu có)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Tắt gradient để tăng tốc
    with torch.no_grad():
        output = model(**inputs)

    last_hidden_state = output.last_hidden_state  # [batch, seq_len, hidden_dim]
    attention_mask = inputs["attention_mask"]

    # Mở rộng attention mask để mean pooling
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()

    # Tính tổng embedding có mask
    sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)

    # Tính số token hợp lệ
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    # Mean pooling
    embedding = sum_embeddings / sum_mask

    return embedding.cpu().numpy()[0].astype("float32")


class PhoBERTCustomEmbeddings(Embeddings):
    # Class Embedding tùy chỉnh để dùng PhoBERT với LangChain.
    def __init__(self):
        # Load model, tokenizer khi khởi tạo
        load_model_components()

    def embed_query(self, text: str) -> List[float]:
        # Tính embedding cho 1 câu truy vấn.
        return get_embedding(text).tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Tính embedding cho nhiều văn bản.
        return [self.embed_query(text) for text in texts]