from ContextRetrievalStore import OllamaMemoryBuilder as OLM
from Util.llm_util import get_embedding_dim_ollama
from Util.util import (read_file,__std_out__)
import controller.config as config

# global log function
def log(*args):
      if config.MODE.lower() != 'dev':
            return
      print(__std_out__(*args))

class Controller(object):

    def __init__(self,mode:str,storage_dir:str):
          """
          Initialize Luci Controller class.

          Args:
              mode: 'dev' or 'prod' if not prod then it will use dev as default mode.
              storage_dir: Directory for persistence
          """
          config.MODE=mode
          # LOAD Instruction for CHAT
          self.core_instruction = read_file(r"D:\Projects\Personal\LLM\Luci\Instructions\crs_handle_instruction.txt")
          log(f"core_instruction loaded:\n ${self.core_instruction}")





