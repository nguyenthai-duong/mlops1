from fastapi import FastAPI, HTTPException
import weaviate
import requests
from llama_index.core import PromptTemplate

# Định nghĩa các URL
# http://<name_svc>.<name_ns>.svc.cluster.local:port/vectorize    -> clusterIP
WEAVIATE_URL = "http://weaviate.weaviate.svc.cluster.local:80"
VECTORIZE_URL = "http://emb-svc.emb.svc.cluster.local:81/vectorize"
LLM_API_URL = "https://nthaiduong83.serveo.net/complete"

app = FastAPI()

def query_rag_llm(query_str, limit=3):
    # Khởi tạo client Weaviate
    client = weaviate.Client(WEAVIATE_URL)

    # Gửi yêu cầu vector hóa câu truy vấn
    text_data = {"text": query_str}
    response = requests.post(VECTORIZE_URL, json=text_data)

    # Kiểm tra phản hồi
    if response.status_code == 200:
        vec = response.json().get("vector")
    else:
        print("Failed to get vector, status code:", response.status_code)
        return None

    # Tìm kiếm trong Weaviate với vector truy vấn
    near_vec = {"vector": vec}
    res = client \
        .query.get("Document", ["content", "_additional {certainty}"]) \
        .with_near_vector(near_vec) \
        .with_limit(limit) \
        .do()

    # Tạo chuỗi ngữ cảnh từ các tài liệu trả về
    context_str = []
    for document in res["data"]["Get"]["Document"]:
        context_str.append("{:.4f}: {}".format(document["_additional"]["certainty"], document["content"]))
        print("{:.4f}: {}".format(document["_additional"]["certainty"], document["content"]))
        print("__________-")
    
    context_str = "\n".join(context_str)

    # Tạo PromptTemplate cho câu hỏi
    template = (
        "We have provided context information below. \n"
        "---------------------\n"
        "{context_str}"
        "\n---------------------\n"
        "Given this information, please answer the question: {query_str}\n"
    )
    qa_template = PromptTemplate(template)
    messages = qa_template.format_messages(context_str=context_str, query_str=query_str)
    prompt = messages[0].content


    response = requests.post(LLM_API_URL, json={"prompt": prompt})

    if response.status_code == 200:
        text_response = response.json().get("response")["text"]
        return text_response
    else:
        print("Failed to get response from LLM, status code:", response.status_code)
        return None

@app.post("/query")
async def query(query_str: str):
    response = query_rag_llm(query_str)
    if response:
        return {"response": response}
    else:
        raise HTTPException(status_code=500, detail="Failed to process the query")

# Chạy ứng dụng FastAPI với Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8005, reload=True)
