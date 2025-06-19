import PyPDF2
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import subprocess
import numpy as np
from opencc import OpenCC

# 文本转换工具
cc_t2s = OpenCC('t2s')  # 繁体 -> 简体
cc_s2t = OpenCC('s2t')  # 简体 -> 繁体



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



def retrieve_top_k_chunks(query, chunks, embeddings, embed_model, k=3):
    query_embedding = embed_model.encode([query])
    sims = cosine_similarity(query_embedding, embeddings)[0]
    top_indices = np.argsort(sims)[-k:][::-1]
    return [chunks[i] for i in top_indices]



def query_ollama(context, question, model_name="llama3"):
    full_prompt = f"根据以下文档内容回答问题：\n\n{context}\n\n问题：{question}"
    command = ["ollama", "run", model_name]
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding='utf-8'  # 避免中文输入报错
    )

    output, error = process.communicate(input=full_prompt)
    if process.returncode == 0:
        return output
    else:
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
