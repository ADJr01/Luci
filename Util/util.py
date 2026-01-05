from ollama import embed

def get_embedding_dim_ollama(model:str):
    result = embed(model=model,input="test query")
    return len(result['embeddings'][0])