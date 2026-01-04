from OllamaMem import Memory, OllamaMemoryBuilder as OLM
import hashlib
OLLAMA_MODEL = 'rnj-1:latest'
EMBEDDING_MODEL = 'qwen3-embedding:0.6b'


if __name__ == '__main__':
    mem = (OLM.OllamaMemoryBuilder()
              .collection_name("test_collection")
              .persist_dir("./faiss_storage")
              .ollama_model(OLLAMA_MODEL)
              .embedding_model(EMBEDDING_MODEL)
              .max_tokens(3000)
              .temperature(0.1314)
              .build())

    mem.set_user_session("rahat.adnan")
    mem.add("I love teal color.",metadata={"feeling":"love","thing":"color"})
    mem.add("I hate Sun",metadata={"feeling":"hate","thing":"sun"})
    res = mem.search("thing i love")
    print(res)


