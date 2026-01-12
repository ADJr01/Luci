import datetime
from langchain_core.documents import Document


class QueryMeta:

    def __init__(self):
        self._query_time:datetime.datetime=datetime.datetime.now()
        self._response_sources:list[str]=[]
        self._response_toolset:list[str]=[]
        self._query_sentiment: str=""
        self._response_sentiment:str=""
        self._response_model:str=""
        self._chat_index:int=0
        self._chat_id:str=""

    def set_chat_id(self,chat_id:str):
        self._chat_id = chat_id
        return self

    def set_query_time(self,query_time:datetime.datetime):
        self._query_time = query_time
        return self

    def set_chat_index(self,chat_index:int):
        self._chat_index = chat_index
        return self

    def add_response_source(self,response_source:str):
        self._response_sources.append(response_source)
        return self
    def remove_response_source(self,response_source:str):
        self._response_sources.remove(response_source)
        return self

    def add_response_toolset(self,response_toolset:str):
        self._response_toolset.append(response_toolset)
        return self

    def set_response_model(self,response_model:str):
        self._response_model=response_model
        return self

    def set_query_sentiment(self,query_sentiment:str):
        self._query_sentiment=query_sentiment
        return self

    def set_response_sentiment(self,response_sentiment:str):
        self._response_sentiment=response_sentiment
        return self

    def __str__(self):
        return f"""
            chat_id: {self._chat_id}\n
            chat_index: {self._chat_index}\n
            response_time: {self._query_time}\n
            response_source: [{", ".join(self._response_sources)}]\n
            response_toolset: [{", ".join(self._response_toolset)}]\n
            query_sentiment: {self._query_sentiment}\n
            response_model: {self._response_model}\n
            response_sentiment: {self._response_sentiment}\n
        """

    def as_meta(self):
        return {
            "chat_id":self._chat_id,
            "chat_index":self._chat_index,
            "response_time":self._query_time,
            "response_source":self._response_sources,
            "response_toolset":", ".join(self._response_toolset),
            "query_sentiment":self._query_sentiment, # -
            "response_sentiment":self._response_sentiment, # -
            "response_model":self._response_model,
            "response_sources":", ".join(self._response_sources),
        }


class Record:

    def __init__(self):
        self._query:str=''
        self._answer:str=''
        self._meta:QueryMeta = QueryMeta()

    def set_query(self,query:str):
        self._query = query
        return self

    def set_answer(self,answer:str):
        self._answer = answer
        return self

    def get_meta(self):
        return self._meta

    def set_meta(self,meta:QueryMeta):
        self._meta = meta
        return self

    def query_str(self):
        return self._query

    def answer_str(self):
        return self._answer

    def __str__(self)->str:
        return f"""
            meta: {self._meta}
            query: {self._query}
            answer: {self._answer}
        """

    def to_doc(self)->Document:
        return Document(
            page_content=f"{self._answer}",
            metadata={**self._meta.as_meta(),"query":self._query},
        )

    def as_dict(self)->dict:
        return {
            "metadata":self.get_meta(),
            "query":self._query,
            "answer":self._answer,
        }