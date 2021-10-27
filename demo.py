import re

if __name__ == "__main__":
    s = '["交易步骤\\[.*？\\]","运行结束，返回值为：\\d+\\""]'
    print(re.findall("\"(.*?[\"])\"", s))
    print(eval(s))
    pass
