import datetime
import time


class QueryMeta:

    def __init__(self):
        self._query_time:datetime.datetime=datetime.datetime.now()
        self._response_sources:list[str]=None
        self._response_toolset:list[str]=None
        self._response_model:str=None
        self._query_sentiment:str=None
        self._response_sentiment:str=None

    def set_query_time(self,query_time:datetime.datetime):
        self._query_time = query_time
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
            response_time: {self._query_time}\n
            response_source: [{", ".join(self._response_sources)}]\n
            response_toolset: [{", ".join(self._response_toolset)}]\n
            query_sentiment: {self._query_sentiment}\n
            response_sentiment: {self._response_sentiment}\n
            
            
        """


class Record:


    def __init__(self):
        self._query:str=''
        self._answer:str=''
        self._meta:QueryMeta=None