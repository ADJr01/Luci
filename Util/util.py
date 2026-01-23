def __std_out__(**text)->str:
    return " ".join(text)



def read_file(path:str):
    with open(path,'r') as f:
        text = ""
        for line in f:
            text += str(line)
        return text