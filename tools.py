from datetime import datetime
import requests
from utilities.utilities import log
import json
from rag import load_chunk, search

_knowledge_chunks = None

def get_current_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def get_city_weather(city):
    url = f"https://wttr.in/{city}?format=3"
    response = requests.get(url, timeout=10)
    response.raise_for_status() # this is raise HTTP error if something goes wrong
    return response.text

def search_knowledge_base(query):
    global _knowledge_chunks
    if _knowledge_chunks is None:
        _knowledge_chunks = load_chunk()

    results = search(query, _knowledge_chunks, top_k=2)
    log(f"Knowledge base search: {query}")

    for r in results:
        log(f"{r['score']:.2f} - {r['source']} - {r['chunk_id']}")

    blocks = []

    for r in results:
        sourceFile = r["source"]
        chunkId = r["chunk_id"]
        chunkText = r["text"]

        header = f"[{sourceFile} #{chunkId}]"
        block = f"{header} \n {chunkText}"
        blocks.append(block)

    return "\n\n".join(blocks)



TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Returns the date and time as a formatted string",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
        },
    },
    {
        "type":"function",
        "function":{
            "name": "get_city_weather",
            "description":"Returns the current weather of a specific City",
            "parameters":{
                "type":"object",
                "properties":{
                    "city":{
                        "type":"string",
                        "description":"City to get weather for"
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type":"function",
        "function":{
            "name":"search_knowledge_base",
            "description": "Search the company's knowledge base for text relevant to a question. "
                            "Use this when answering questions about company policies or other internal documents. "
                            "Returns the most relevant matching content and its sources; base your answer on that content.",

            "parameters":{
                "type":"object",
                "properties":{
                    "query":{
                        "type":"string",
                        "description":"The user's question rewritten as a short, self-contained search query."
                    }
                },
                "required":["query"],
            }
        }

    }


]

FUNCTION_NAME_TO_FUNCTION_MAP = {
    "get_current_time": get_current_time,
    "get_city_weather": get_city_weather,
    "search_knowledge_base": search_knowledge_base,
}

def run_tool(tool_call):
    func_name = tool_call.function.name
    func_args={}
    
    try:
        func_args = json.loads(tool_call.function.arguments or "{}")
        fn = FUNCTION_NAME_TO_FUNCTION_MAP[func_name]
        log(f"🗝️ {func_name} {func_args}")
        result = fn(**func_args)
        log(f"  -> {result}")
    except Exception as e:
        result = f"Error while running {func_name}: {e}"

    return result
