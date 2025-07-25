<<<<<<< HEAD
from flask import Flask, request, jsonify
=======
import PyPDF2
>>>>>>> b68ba2e77bf90b691144b988de751c75581def6d
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import subprocess
import numpy as np
from opencc import OpenCC
<<<<<<< HEAD
import json

app = Flask(__name__)
=======
>>>>>>> b68ba2e77bf90b691144b988de751c75581def6d

# 文本转换工具
cc_t2s = OpenCC('t2s')  # 繁体 -> 简体
cc_s2t = OpenCC('s2t')  # 简体 -> 繁体

<<<<<<< HEAD
# === 載入 JSON 並轉換為段落 ===
def read_json_text(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def flatten_json_entries(data):
    chunks = []
    for entry in data:
        block = f"膚質類型：{entry['skin_type']}\n"
        for c in entry.get("recommended_courses", []):
            block += f"療程：{c['name']}\n功效：{'、'.join(c.get('goals', []))}\n"
        for p in entry.get("products", []):
            block += f"產品：{p['name']}（{p['step']}）\n特色：{'；'.join(p.get('features', []))}\n"
        block += f"推薦話術：{entry.get('reply_template', '')}"
        chunks.append(block)
    return chunks

# === 文字向量處理 ===
def embed_chunks(chunks, model):
    return model.encode(chunks)

=======


def read_pdf_text(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()



def split_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks



def embed_chunks(chunks, model):
    return model.encode(chunks)



>>>>>>> b68ba2e77bf90b691144b988de751c75581def6d
def retrieve_top_k_chunks(query, chunks, embeddings, embed_model, k=3):
    query_embedding = embed_model.encode([query])
    sims = cosine_similarity(query_embedding, embeddings)[0]
    top_indices = np.argsort(sims)[-k:][::-1]
    return [chunks[i] for i in top_indices]

<<<<<<< HEAD
# === 聊天記憶 + 提問模型 ===
def query_ollama(context, question, history, model_name="llama3"):
    history_block = "\n".join([f"使用者：{q}\n助理：{a}" for q, a in history])

    full_prompt = f"""
你是一位專業的皮膚保養顧問，擅長用清楚、溫柔且條列式的方式提供建議。
請根據【參考資料】與【歷史對話】，針對「{question}」這個問題，列出 3～5 點清楚的建議。

【歷史對話】
{history_block}

【參考資料】
{context}

重要規則：
- 回答內容「必須完整使用繁體中文」，不可出現英文單字、片語或表情符號
- 每一點請簡潔扼要，以 1. 2. 3. 條列呈現，每點不超過兩行
- 不需要開場白、感嘆語或總結語
- 語氣要親切、具體
"""

=======


def query_ollama(context, question, model_name="llama3"):
    full_prompt = f"根据以下文档内容回答问题：\n\n{context}\n\n问题：{question}"
>>>>>>> b68ba2e77bf90b691144b988de751c75581def6d
    command = ["ollama", "run", model_name]
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
<<<<<<< HEAD
        encoding='utf-8'
=======
        encoding='utf-8'  # 避免中文输入报错
>>>>>>> b68ba2e77bf90b691144b988de751c75581def6d
    )

    output, error = process.communicate(input=full_prompt)
    if process.returncode == 0:
        return output
    else:
<<<<<<< HEAD
        return f"模型錯誤：\n{error}"

# === 初始化模型與資料 ===
print("loading skincare data...")
data = read_json_text("C:\\Data\\skincare_data (1).json")
chunks = flatten_json_entries(data)
print("embedding data...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
chunk_embeddings = embed_chunks(chunks, embed_model)
history = []  # 多輪對話歷史紀錄

# === Flask 聊天端點 ===
@app.route("/chat", methods=["POST"])
def chat():
    req = request.get_json()
    question_trad = req.get("question", "")
    if not question_trad:
        return jsonify({"error": "請提供問題"}), 400

    question_simp = cc_t2s.convert(question_trad)
    top_chunks = retrieve_top_k_chunks(question_simp, chunks, chunk_embeddings, embed_model, k=3)
    context = "\n---\n".join(top_chunks)
    answer_simp = query_ollama(context, question_simp, history)
    answer_trad = cc_s2t.convert(answer_simp)

    history.append((question_trad, answer_trad))
    return jsonify({"response": answer_trad})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
=======
        return f"模型错误：\n{error}"



def main():
    # 设置 PDF 路径
    pdf_path = "C:\\Data\\皮肤保养品使用文章.pdf"
    print("reading PDF...")

    # 加载并切分文档
    full_text = read_pdf_text(pdf_path)
    chunks = split_text(full_text)

    print("loading embedding model...")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    chunk_embeddings = embed_chunks(chunks, embed_model)

    print("初始化完成，可以开始提问（输入 q 退出）")

    
    while True:
        question_trad = input("請輸入您的問題（繁體中文）：")
        if question_trad.lower() in ["exit", "quit", "q"]:
            print("再見！")
            break

        # Step 1: 用户输入繁体 -> 简体
        question_simp = cc_t2s.convert(question_trad)

        # Step 2: 用简体问题进行语义搜索
        top_chunks = retrieve_top_k_chunks(question_simp, chunks, chunk_embeddings, embed_model, k=3)
        context = "\n---\n".join(top_chunks)

        print("搜尋中，請稍候...\n")

        # Step 3: 仍使用简体提问给模型（因为上下文和问题一致）
        answer_simp = query_ollama(context, question_simp)

        # Step 4（可选）: 输出转换为繁体
        answer_trad = cc_s2t.convert(answer_simp)

        print("回答：\n", answer_trad)


if __name__ == "__main__":
    main()
>>>>>>> b68ba2e77bf90b691144b988de751c75581def6d
