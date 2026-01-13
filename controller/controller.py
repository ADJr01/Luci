from ContextRetrievalStore import OllamaMemoryBuilder as OLM
from Util.llm_util import get_embedding_dim_ollama
from helper import (
      EMBEDDING_MODEL,
      OLLAMA_MODEL,
)
class Controller(object):
    def __init__(self):
        self.mem = (OLM.OllamaMemoryBuilder()
              .collection_name("test_collection")
              .persist_dir("./store")
              .max_tokens(3000)
              .embedding_dims(dims=get_embedding_dim_ollama(EMBEDDING_MODEL))
              .ollama_model(OLLAMA_MODEL)
              .embedding_model(EMBEDDING_MODEL)
              .max_tokens(3000)
              .temperature(0.1314)
              .build())
        # mem.set_user_session("rahat.adnan")
        # # mem.add(text="i love to play football",metadata={"source":"user_preference.txt"})
        # res = mem.query(query="things i like")
        # for i in res['results']:
        #     print(f"ans: {i['memory']}[score: {i['score']}]")
        # mem.end_user_session()
