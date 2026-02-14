from ContextRetrievalStore import OllamaMemoryBuilder as OLM
from Util.llm_util import get_embedding_dim_ollama
from Util.util import (read_file,__std_out__)
import controller.config as config
from controller.CastOllama.CastOllama import CastOllama
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
          self.llm = None
          self.core_instruction = read_file(r"D:\Projects\Personal\LLM\Luci\Instructions\crs_handle_instruction.txt")
          log(f"core_instruction loaded:\n ${self.core_instruction}")
          self.__load__OLLAMA_MODEL() #loaded ollama model


    def __load__OLLAMA_MODEL(self):
        self.llm = (
            CastOllama.builder()
            .set_model(config.OLLAMA_MODEl)
            .set_host("http://localhost:11434")
            .set_temperature(0.7)
            .set_context_length(32768)
            .set_max_tokens(1024)
            .set_system_prompt("You are a helpful assistant.")
            .enable_streaming(False)
            .enable_thinking(False)
            .enable_web_search(False)  # set api_key= for live search
            .set_log_level("INFO")
            .build()
        )





