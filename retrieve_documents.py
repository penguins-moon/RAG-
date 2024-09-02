import pdfplumber
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import faiss
import re
import os
import sys
import json
import torch
from zhipu import call_response
import numpy as np
import jieba
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

os.chdir(sys.path[0])

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # 转换为 Python 列表
        return super(NumpyEncoder, self).default(obj)

# 加载PDF文档并提取文本
def load_pdf(file_path):
    with pdfplumber.open(file_path) as pdf:
        full_text = []
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:  # 确保页面有文本
                full_text.append(page_text)
    full_text = ''.join(full_text)
    return full_text


def match3(text):
    pattern = r"第[一二三四五六七八九十][章节] 管理层讨论与分析(.*?)第[一二三四五六七八九十][章节]"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:  
       return ''.join(matches)  
    else:
        return text

def chunk_by_langchain(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,
        separators=[
        "\n\n",
        "\n",
        " ",
        ".",
        ",",
        "\u200B",  # Zero-width space
        "\uff0c",  # Fullwidth comma
        "\u3001",  # Ideographic comma
        "\uff0e",  # Fullwidth full stop
        "\u3002",  # Ideographic full stop
        "",
        ],
    )
    chunks = text_splitter.create_documents([text])
    return [chunk.page_content for chunk in chunks]


def build_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index


class knowledge:
    def __init__(self, address):
        self.knowledgebase = json.load(open(address, encoding='utf-8'))
        self.key_cut = [jieba.lcut(x) for x in self.knowledgebase.keys()]
        self.bm25 = BM25Okapi(self.key_cut)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = SentenceTransformer('./xiaobu',device=device)
        # self.model = SentenceTransformer("lier007/xiaobu-embedding-v2",device=device) #("BAAI/bge-large-zh-v1.5",device=device)
    
    def update(self):
        self.key_cut = [jieba.lcut(x) for x in self.knowledgebase.keys()]
        self.bm25 = BM25Okapi(self.key_cut)

    def get_knowldegebase(self,prompt):
        doc_scores = self.bm25.get_scores(jieba.lcut(prompt))
        top_indice = doc_scores.argmax()
        name_list = list(self.knowledgebase.keys())
        name = name_list[top_indice]
        print(name)
        return np.array(self.knowledgebase[name]['embeddings']), self.knowledgebase[name]['text']

    def generate(self,query):
        embeddings, chunks = self.get_knowldegebase(query)
        index = build_index(embeddings)
        context = self.retrieve(query, index, chunks, embeddings, k=20)
        input_text = f"""
        Context: {context}
        Question: {query}
        requirement:你是一个公司年报分析助手，请阅读年报材料和表格，以专业化的口吻回答问题, 视问题情况尽可能详实地回答问题\
        Answer Template:您好！根据提供的信息，... ...
        """
        return call_response(input_text)
    
    def get_keys(self):
        a = list(self.knowledgebase.keys())
        # print(a)
        return a
    
    def embed_text(self, text_blocks):
        embeddings = self.model.encode(text_blocks, show_progress_bar=False)
        return embeddings
    
    def retrieve(self, query, index, text_blocks, embeddings, k=10):
        query_embedding = self.embed_text([query])[0]
        # Example of checking dimensions
        print("Dimension of index:", index.d)
        print("Dimension of query embedding:", query_embedding.shape) 
        _ , indices = index.search(query_embedding.reshape(1, -1), 50)
        
        retrieved_texts = [text_blocks[i] for i in indices[0]]
        retrieved_embeddings = np.array([embeddings[i] for i in indices[0]])
        # 计算余弦相似度
        similarities = cosine_similarity(query_embedding.reshape(1,-1), retrieved_embeddings).flatten()
        
        top_k_indices = np.argsort(similarities)[-k:]
        top_k_texts = [] # index range?
        for i in top_k_indices:
            if 1 <= i < k-1:
                top_k_texts.append(''.join(retrieved_texts[i-1:i+2]))
            else:
                top_k_texts.append(retrieved_texts[i]) #上下文

        return top_k_texts
    
    def update_json_file(self, file_path, new_content):
        self.knowledgebase.update(new_content)
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(self.knowledgebase, file, ensure_ascii=False, cls = NumpyEncoder,indent=4)        
        print('Updated successfully')


    def upload_file_cli(self, address='./knowledgebase.json'):
        location = input('输入本地pdf地址: ')
        name = input('输入索引名：')
        print('extracting...')
        text = load_pdf(location)
        selected_text = match3(text)
        chunks = chunk_by_langchain(selected_text)
       
        for chunk in chunks:
            chunk = chunk.replace('\n','')
            print(chunk)
            print('\n')
        print(len(chunks))
        print('embedding...')
        embeddings = self.embed_text(chunks)
        new_content = {
            f"{name}":{ 
                "embeddings":embeddings,
                "text": chunks
                }
        }
        print("saving...")
        self.update_json_file(address, new_content)


    def upload_files(self, file, address='./knowledgebase.json'):
        file_path = file.name
        file_name = os.path.basename(file_path) 
        print('extracting...')
        selected_text = match3(load_pdf(file_path))
        chunks = chunk_by_langchain(selected_text)
        for chunk in chunks:
            chunk = chunk.replace('\n','')
        print(len(chunks))
        print('embedding...')
        embeddings = self.embed_text(chunks)
        new_content = {
            f"{file_name[:-4]}":{ 
                "embeddings":embeddings,
                "text": chunks
                }
        }
        print("saving...")
        self.update_json_file(address, new_content)
        self.update()
        keys = self.get_keys()
        return pd.DataFrame(keys, columns=['Available files'])
    
    def launch(self):
        address = input('输入本地pdf地址：')
        full_text = load_pdf(address)
        text = ''.join(full_text)
        print('chunking...')
        chunks = chunk_by_langchain(text) 
        print('embedding...')
        embeddings = self.embed_text(chunks)
        print('indexing...')
        index = build_index(embeddings)
        print('file loaded successfully')
        cnt = 0
        while cnt < 100:
            query = input("\nuser: ")
            if query == 'quit':
                break  
            context = self.retrieve(query, index, chunks, embeddings, k=20)
            input_text = f"""
            Context: {context}
            Question: {query}
            requirement:你是一个公司年报分析助手，请阅读年报材料和表格，以专业化的口吻回答问题\
            Answer Template:您好！根据提供的信息，... ...
            """
            print(input_text)
            answer = call_response(input_text) 
            if answer:
                print("\nAI:", answer)
            else:
                print("\nAI: Sorry, I was unable to process your request.")
            cnt += 1


if __name__ == '__main__':
    Base = knowledge('./knowledgebase.json')
    Base.upload_file_cli()




