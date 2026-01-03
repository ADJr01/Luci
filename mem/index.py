from mem0 import Memory

def mem_ollama(collection_name:str,persist_dir:str,ollama_model:str,embedding_model:str,max_tokens:int=3000,temperature:int=0,distance_metric:str="eucladian",ollama_base_url:str="http://localhost:11434/"):
    mem_conf = {
        "vector_store":{
            "provider":"faiss",
            "config":{
                "collection_name":collection_name,
                "path":persist_dir,
                "distance_strategy":distance_metric
            }
        },
        "llm":{
            "provider":"ollama",
            "config":{
                "model":ollama_model,
                "temperature":temperature,
                "max_tokens":max_tokens,
                "ollama_base_url":ollama_base_url
            }
        },
        "embedder":{
            "provider":"ollama",
            "config":{
                "model":embedding_model,
                "ollama_base_url":ollama_base_url

            }
        }
    }
