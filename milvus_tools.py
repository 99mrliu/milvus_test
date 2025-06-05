from typing import List, Dict, Any
from pymilvus import connections, utility
from pymilvus import FieldSchema, CollectionSchema, DataType, Collection
from pymilvus import model
import time
import re
import chardet
import os
from docx import Document
from mcp.server.fastmcp import FastMCP



def prepareData(path):
    """
    遍历指定路径下的文件，对于 .docx、.md 及其他文件，读取内容并做简单清洗后直接返回 file_name 和完整 content。
    """
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        try:
            if file.lower().endswith('.docx'):
                doc = Document(file_path)
                # 提取所有段落内容，并去除制表符、多余空格
                content = ' '.join([para.text.replace('\t', ' ').strip() for para in doc.paragraphs if para.text.strip()])
                content = re.sub(r'[\n\t\s]+', ' ', content)  # 去掉换行符和多余空格
                content = re.sub(r'[^\w\s]', '', content)      # 去掉所有特殊字符
                content = content.replace(" ", "")             # 去掉所有空格
                yield (file, content)

            elif file.lower().endswith('.md'):
                with open(file_path, 'rb') as f:
                    raw_data = f.read()
                    encoding = chardet.detect(raw_data)['encoding']
                    content = raw_data.decode(encoding or 'utf-8', errors='ignore')
                    content = re.sub(r'[\n\t\s]+', ' ', content)  # 去掉换行符和多余空格
                    content = re.sub(r'[^\w\s]', '', content)      # 去掉所有特殊字符
                    content = content.replace(" ", "")             # 去掉所有空格
                    yield (file, content)

            else:
                with open(file_path, 'rb') as f:
                    raw_data = f.read()
                    encoding = chardet.detect(raw_data)['encoding']
                    content = raw_data.decode(encoding or 'utf-8', errors='ignore')
                    content = re.sub(r'[\n\t\s]+', ' ', content)  # 去掉换行符和多余空格
                    content = re.sub(r'[^\w\s]', '', content)      # 去掉所有特殊字符
                    content = content.replace(" ", "")             # 去掉所有空格
                    yield (file, content)

        except Exception as e:
            print(f"无法处理文件 {file}: {e}")
            continue


def getEmbedding(text):
    ef = model.DefaultEmbeddingFunction()
    embedding = ef([text])[0]  # 传入一个文本列表并获取第一个结果
    return embedding



class MilvusImporter:
    def __init__(self, uri: str, token: str):
        self.uri = uri
        self.token = token

    def connect_to_milvus(self):
        try:
            connections.connect(
                uri=self.uri,
                token=self.token
            )
            print("已连接到 Milvus 集群")
            return True
        except Exception as e:
            print(f"连接到 Milvus 集群失败: {e}")
            return False

    def import_data(self, data_path: str, collection_name: str, dimension: int = 768) -> bool:
        if not self.connect_to_milvus():
            return False

        try:
            # 创建集合
            if utility.has_collection(collection_name):
                utility.drop_collection(collection_name)
            collection = self.create_collection(collection_name, dimension)

            # 创建索引
            index_params = {
                'index_type': 'IVF_FLAT',
                'metric_type': 'L2',
                'params': {'nlist': 1024}
            }
            collection.create_index(field_name="embedding", index_params=index_params)

            # 插入数据
            id = 0
            for file_name, content in prepareData(data_path):
                embedding = getEmbedding(content)
                self.insert_data(collection, id, file_name, content, embedding)
                id += 1
                time.sleep(2)  # 确保插入操作完成

            print("数据导入完成")
            return True

        except Exception as e:
            print(f"数据导入失败: {e}")
            return False

    def create_collection(self, collection_name: str, dimension: int):
        # 定义字段
        fields = [
            FieldSchema(name='id', dtype=DataType.INT64, description='Ids', is_primary=True, auto_id=False),
            FieldSchema(name='file_name', dtype=DataType.VARCHAR, description='File Name', max_length=256),
            FieldSchema(name='text', dtype=DataType.VARCHAR, description='Text Content', max_length=65535),
            FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR, description='Embedding vectors', dim=dimension)
        ]

        # 创建集合 schema
        schema = CollectionSchema(fields=fields, description='None')

        # 创建集合
        collection = Collection(name=collection_name, schema=schema)

        return collection

    def insert_data(self, collection, id, file_name, text, embedding):
        text = text.replace('\n', ' ').replace('\t', ' ').strip()  # 去掉换行符和制表符
        insert_data = [
            [id],  # id
            [file_name],  # file_name
            [text],  # text
            [embedding]  # embedding
        ]
        collection.insert(insert_data)

