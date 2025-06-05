from typing import List, Dict, Any
from pymilvus import connections, Collection
from pymilvus import model



class SearchResult:
    """Class to store search results."""
    def __init__(self, id: int, filename: str, text: str):
        self.id = id
        self.filename = filename
        self.text = text

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "id": self.id,
            "filename": self.filename,
            "text": self.text
        }

def getEmbedding(text):
    ef = model.DefaultEmbeddingFunction()
    embedding = ef([text])[0]  # 传入一个文本列表并获取第一个结果
    return embedding

class MilvusSearchTool:
    def __init__(self, collection_name: str, uri: str, token: str):
        self.collection_name = collection_name
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

    def search_similar_texts(self, query_text: str, top_k: int = 2) -> List[Dict[str, Any]]:
        if not self.connect_to_milvus():
            return []

        try:
            # 加载集合
            collection = Collection(self.collection_name)
            collection.load()

            # 生成查询向量
            query_embedding = getEmbedding(query_text)

            # 检索参数
            search_params = {
                "metric_type": "L2",
                "params": {"nprobe": 10}
            }

            # 执行检索并返回 text 和 file_name 字段
            results = collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                output_fields=["text", "file_name"]  # 指定返回的字段
            )

            search_results = []
            for hits in results:
                for hit in hits:
                    text = hit.entity.get('text')
                    file_name = hit.entity.get('file_name')
                    id = hit.id

                    search_result = SearchResult(id, file_name, text)
                    search_results.append(search_result.to_dict())

            return search_results

        except Exception as e:
            print(f"检索失败: {e}")
            return []
