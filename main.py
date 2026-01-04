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
    rahat_token = hashlib.sha256("rahat.adnan:my-secret-key".encode()).hexdigest()
    mem.set_user("rahat.adnan", rahat_token)
    mem.add("I am Rahat Adnan.I am 29 years Old.I live in Dhaka Cantonment")


