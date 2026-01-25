from ContextRetrievalStore import OllamaMemoryBuilder as OLM
from Util.llm_util import get_embedding_dim_ollama
from Util.util import (read_file,__std_out__)
from config import (
      EMBEDDING_MODEL,
      OLLAMA_MODEL,
      CHAT_MODEl,
      MODE
)
def log(**args):
      if MODE is 'DEV':
            print(__std_out__(**args))

class Controller(object):
    def __init__(self):
          # LOAD Instruction for CHAT
          self.core_instruction = read_file(r"D:\Projects\Personal\LLM\Luci\Instructions\crs_handle_instruction.txt")


