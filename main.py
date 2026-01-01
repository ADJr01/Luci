# This is a sample Python script.
from CRS.Record import Record
import datetime
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

if __name__ == '__main__':
    record = Record()
    meta = record.get_meta()
    meta.set_chat_id("chat_id_2025-30-12_10:30").add_response_toolset("search_google").set_query_time(datetime.datetime.now()).set_chat_index(1).set_response_model("deepseek-r1:8b").set_query_sentiment("positive").add_response_source("https://www.techpowerup.com/cpu-specs/ryzen-7-5700g.c2472")
    record.set_meta(meta).set_query("I am using AMD Ryzen 7 5700G.How much thread it has?").set_answer(f"""AMD Ryzen 7 5700G processor has 8 core thread and 16 thread with Zen 3 (Cezanne) architecture.""")
    print(record.to_doc())
#page_content='AMD Ryzen 7 5700G processor has 8 core thread and 16 thread with Zen 3 (Cezanne) architecture.' metadata={'chat_id': 'chat_id_2025-30-12_10:30', 'chat_index': 1, 'response_time': datetime.datetime(2026, 1, 1, 2, 35, 11, 813137), 'response_source': ['https://www.techpowerup.com/cpu-specs/ryzen-7-5700g.c2472'], 'response_toolset': 'search_google', 'query_sentiment': 'positive', 'response_sentiment': '', 'response_model': 'deepseek-r1:8b', 'response_sources': 'https://www.techpowerup.com/cpu-specs/ryzen-7-5700g.c2472', 'query': 'I am using AMD Ryzen 7 5700G.How much thread it has?'}