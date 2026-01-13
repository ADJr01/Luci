from ContextRetrievalStore.Record import Record


if __name__ == '__main__':
    record = Record()
    meta = record.get_meta()
    (meta.set_chat_id('2026_01_13_66srt')
     .set_chat_index(12)
     .set_response_model("gpt-oss:20b")).set_query_sentiment("")








