import faiss
import pickle
import numpy as np
from typing import List, Any
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from . import config
from .embeddings import PhoBERTCustomEmbeddings
from pydantic import Field

# Hàm load index FAISS và metadata
def load_faiss_and_metadata():
    """
    Tải FAISS index từ file và metadata đi kèm (pickle).
    Trả về (index, metadata).
    """
    print("Load file index và metadata")
    # Đọc index FAISS từ đường dẫn cấu hình
    index = faiss.read_index(config.index_path)

    # Đọc metadata (dưới dạng list, dict) từ file pickle
    with open(config.metadata_path, "rb") as f:
        metadata = pickle.load(f)

    return index, metadata


# ---------- Retriever tùy chỉnh dùng FAISS + PhoBERT embeddings ----------
class FAISSCustomRetriever(BaseRetriever):
    """
    Retriever tuỳ chỉnh kế thừa BaseRetriever của LangChain.
    Sử dụng FAISS để tìm các document tương tự dựa trên embedding từ PhoBERT.
    """

    # Các field pydantic để định nghĩa thuộc tính (dùng khi khởi tạo model pydantic)
    index: Any = Field(...)           # FAISS index object
    # Đổi tên thành doc_metadata để tránh xung đột tên biến 'metadata' của hệ thống
    doc_metadata: Any = Field(...)    # Metadata tương ứng với các vectors trong index (list/dict)
    embeddings: PhoBERTCustomEmbeddings = Field(...)  # client embedding (class custom)
    k: int = Field(default=config.top_k_retrieval)   # số lượng trả về (top-k)

    def __init__(self, index, metadata, embeddings_client, k = config.top_k_retrieval, **kwargs):
        """
        Khởi tạo lớp retriever — truyền index, metadata, embeddings client và k.
        Gọi super().__init__ để BaseRetriever có thể xử lý các tham số nếu cần.
        """
        super().__init__(
            index=index,
            doc_metadata=metadata,
            embeddings=embeddings_client,
            k=k,
            **kwargs
        )

    def _get_relevant_documents(self, query: str, **kwargs: Any) -> List[Document]:
        """
        Đồng bộ: nhận 1 query string, tính vector bằng embeddings.embed_query,
        tìm kiếm trong FAISS index, sau đó map kết quả sang list[Document].
        """
        # Tính embedding cho query (list hoặc numpy array 1-D)
        query_vector = self.embeddings.embed_query(query)

        # FAISS expect float32 numpy array with shape (n, dim)
        D, I = self.index.search(np.array([query_vector]).astype("float32"), k=self.k)

        docs = []
        # I[0] là danh sách index của các vector gần nhất cho query đầu tiên
        for i, idx in enumerate(I[0]):
            # Lấy item metadata tương ứng
            metadata_item = self.doc_metadata[idx]

            # Tạo nội dung hiển thị (ở đây bạn ghép trường 'ten_benh' và 'trieu_chung')
            content = f"Tên bệnh: {metadata_item['ten_benh']}. Triệu chứng {metadata_item['trieu_chung']}"

            # Tạo Document (page_content + metadata): LangChain Document object
            docs.append(Document(
                page_content=content,
                metadata={"source_id": metadata_item["id"]}
            ))

        return docs

    async def aget_relevant_documents(self, query: str, **kwargs: Any) -> List[Document]:
        """
        Phiên bản bất đồng bộ: ở đây chỉ gọi lại hàm đồng bộ để đơn giản.
        Nếu muốn non-blocking thật sự, có thể đưa việc tính embedding/search
        vào threadpool hoặc implement I/O async tương ứng.
        """
        return self._get_relevant_documents(query, **kwargs)


# Hàm tiện ích để khởi tạo retriever
def initialize_retriever():
    """
    Tải index + metadata, khởi tạo embeddings client và trả về một instance
    của FAISSCustomRetriever đã sẵn sàng sử dụng.
    """
    index, metadata = load_faiss_and_metadata()
    embeddings_client = PhoBERTCustomEmbeddings()
    return FAISSCustomRetriever(index, metadata, embeddings_client)
