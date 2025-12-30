# This is a sample Python script.
from CRS.Record import Record
import datetime
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

if __name__ == '__main__':
    record = Record()
    meta = record.get_meta()
    meta.set_chat_id("chat_id_2025-30-12_10:30").set_query_time(datetime.datetime.now()).set_chat_index(1).set_response_model("deepseek-r1:8b").set_query_sentiment("positive").add_response_source("sky.pdf")
    record.set_meta(meta).set_query("The Sky is ").set_answer("Blue")
    print(record.to_doc())