from mcp.server.fastmcp import FastMCP
from mcp.server.sse import SseServerTransport
from starlette.applications import Starlette
from starlette.routing import Mount, Route
import uvicorn
import httpx
from milvus_tools import MilvusImporter
from typing import List, Dict, Any
from rag_tools import MilvusSearchTool,SearchResult

mcpserver=FastMCP(name='rag-system') # handle session



@mcpserver.tool(name='data_import',description='import data to milvus')
async def dataimport(data_path: str, collection_name: str):
    """
    导入数据到 Milvus 集群。
    Args:
        data_path: 数据文件夹路径。
        collection_name: Milvus 集合名称。

    Returns:
        Dict[str, Any]: 导入结果。
    """
    uri = ""
    token = ""
    importer = MilvusImporter(uri, token)
    success = importer.import_data(data_path, collection_name)
    return {
        "success": success,
        "message": "数据导入成功" if success else "数据导入失败"
    }

@mcpserver.tool(name='milvus_search',description='Retrieve similar text in Milvus.')
async def milvus_search(query: str,collection:str) -> List[Dict[str, Any]]:
    """
    在 Milvus 中检索相似文本。

    Args:
        query: 检索查询文本。
        collection: 集合名字

    Returns:
        List[Dict[str, Any]]: 检索结果的字典列表。
    """
    collection_name =  collection
    uri = ""
    token = ""

    tool = MilvusSearchTool(collection_name, uri, token)
    return tool.search_similar_texts(query)


if __name__ == '__main__':
    mcpserver.run(transport='stdio')
