from ContextRetrievalStore import OllamaMemoryBuilder as OLM
from Util.llm_util import get_embedding_dim_ollama
from Util.util import read_file
from helper import (
      EMBEDDING_MODEL,
      OLLAMA_MODEL,
      CHAT_MODEl
)
class Controller(object):
    def __init__(self):
          # LOAD Instruction for CHAT
          self.core_instruction = read_file(r"D:\Projects\Personal\LLM\Luci\Instructions\crs_handle_instruction.txt")

