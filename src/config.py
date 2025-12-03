# Đường dẫn tới mô hình PhoBERT đã fine-tuned cho lĩnh vực y tế
model_path = "model\\phobert-medical-final"

# Đường dẫn tới file FAISS index dùng để truy xuất triệu chứng y khoa
index_path = "vector_store\\medical_symptoms.index"

# Đường dẫn tới file metadata (thông tin mô tả tài liệu) đi kèm FAISS index
metadata_path = "vector_store\\medical_symptoms.pkl"

# Kích thước vector embedding của PhoBERT (thường là 768 với bản base)
embedding_dim = 768

# Độ dài tối đa khi token hóa văn bản trước khi đưa vào model
max_length = 256

# Số lượng tài liệu cần truy xuất (top-k) cho mỗi truy vấn
top_k_retrieval = 5