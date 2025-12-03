import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter
from dotenv import load_dotenv

# Import module nội bộ
from .retriever import initialize_retriever

# 1. Cấu hình & Khởi tạo
load_dotenv()
app = FastAPI(title="Medical RAG Chatbot API")

# Cấu hình CORS (Để frontend có thể gọi API nếu chạy port khác)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Khởi tạo Logic RAG (Chỉ chạy 1 lần khi start server)
print("--- Đang khởi tạo mô hình AI & Vector DB ---")
try:
    # Khởi tạo LLM của Google Gemini
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.1  # Giảm randomness, tăng độ chính xác
    )

    # Khởi tạo retriever
    retriever = initialize_retriever()

    # Hàm format tài liệu được truy xuất thành một chuỗi dài
    def format_docs(docs):
        return "\n\n---\n\n".join(doc.page_content for doc in docs)

    # Prompt cho RAG
    template = """
    Bạn là một trợ lý y tế thông minh. Nhiệm vụ của bạn là chẩn đoán sơ bộ dựa trên các triệu chứng được cung cấp.
    Sử dụng NGỮ CẢNH TRUY XUẤT (Context) là các triệu chứng liên quan để đưa ra câu trả lời chính xác, bằng tiếng Việt.

    NGỮ CẢNH TRUY XUẤT:
    {context}

    Câu hỏi của người dùng: {question}

    Phân tích và trả lời:
    """

    # Tạo prompt dạng template
    prompt = ChatPromptTemplate.from_template(template)

    # Xây dựng pipeline RAG
    rag_chain = (
        {
            # Lấy "question" đưa vào retriever vào format docs
            "context": itemgetter("question") | retriever | format_docs,

            # Lấy lại question để đưa vào prompt
            "question": itemgetter("question")
        }
        # Đưa vào prompt template
        | prompt
        # Gửi qua LLM Gemini
        | llm
    )
    print("--- Hệ thống đã sẵn sàng! ---")
except Exception as e:
    print(f"Lỗi khởi tạo: {e}")
    rag_chain = None

# 3. Định nghĩa Data Model cho API
class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str

# 4. API Endpoint
@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    if not rag_chain:
        raise HTTPException(status_code=500, detail="Hệ thống chưa khởi tạo xong.")
    
    try:
        # Gọi RAG Chain
        response = rag_chain.invoke({"question": req.question})
        return ChatResponse(answer=response.content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 5. Serve Frontend (Để truy cập localhost:8000 là ra web luôn)
# Mount thư mục frontend để chạy file html
import os
from pathlib import Path

# Lấy đường dẫn tuyệt đối đến thư mục frontend
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

if os.path.exists(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

# Hàm main để chạy debug
if __name__ == "__main__":
    uvicorn.run("src.app:app", host="0.0.0.0", port=8000, reload=True)