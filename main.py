from OllamaMem import Memory, OllamaMemoryBuilder as OLM
import hashlib
from Util.util import get_embedding_dim_ollama
OLLAMA_MODEL = 'rnj-1:latest'
EMBEDDING_MODEL = 'qwen3-embedding:0.6b'
RERANKING_MODEL = 'dengcao/Qwen3-Reranker-4B:Q8_0'


if __name__ == '__main__':
    mem = (OLM.OllamaMemoryBuilder()
              .collection_name("test_collection")
              .persist_dir("./store")
              .max_tokens(3000)
              .embedding_dims(dims=get_embedding_dim_ollama(EMBEDDING_MODEL))
              .ollama_model(OLLAMA_MODEL)
              .embedding_model(EMBEDDING_MODEL)
              .re_ranking_model(RERANKING_MODEL)
              .max_tokens(3000)
              .temperature(0.1314)
              .build())

    mem.set_user_session("rahat.adnan")

    res = mem.search("things i like")
    for i in res['results']:
        print(f"ans: {i['memory']}[score: {i['score']}]")
    mem.end_user_session()






