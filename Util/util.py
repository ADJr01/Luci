def log(**text)->str:
    str = " ".join(text)
    return str


def read_file(path:str):
    with open(path,'r') as f:
        text = ""
        for line in f:
            text += str(line)
        return text