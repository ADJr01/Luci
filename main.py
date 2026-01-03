from OllamaMem import Memory, OllamaMemoryBuilder as OLM

OLLAMA_MODEL = 'rnj-1:latest'
EMBEDDING_MODEL = 'qwen3-embedding:0.6b'


if __name__ == '__main__':
    mem = (OLM.OllamaMemoryBuilder()
    .collection_name("test_collection")
    .ollama_model(OLLAMA_MODEL)
    .embedding_model(EMBEDDING_MODEL)
    .enable_authentication(False)
    .max_tokens(1500).temperature(0.1314).build())
    mem.set_user("rahat.adnan")
    mem.add("I am Rahat Adnan.I am 29 years Old.I live in Dhaka Cantonment")
    rs = mem.search("do you know my Name")
    print(rs)

