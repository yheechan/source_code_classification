def record_data_info(class_nm, txt_cnt):
    with open('../data/info.txt', 'a') as f:
        f.write(class_nm + ': ' + str(txt_cnt) + '\n')

def record_nonAscii_txt(fn):
    with open('../data/nonAsciiTxt.txt', 'a') as f:
        f.write(fn + '\n')

def print_nonAscii_info(nonAsciiDict):
    with open('../data/nonAsciiInfo.txt', 'a') as f:
        for key in nonAsciiDict:
            f.write(key + '\t : ' + str(nonAsciiDict[key]) + '\n')

def record_ch2idx(ch2idx):
    with open('../data/ch2idx.txt', 'a') as f:
        for key in ch2idx:
            f.write(key + '\t : ' + str(ch2idx[key]) + '\n')