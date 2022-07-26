import os
import info_recorder
import pandas as pd

def get_df():
    data_path = '../data/code25/'
    class_dir_list = [os.listdir(data_path)]
    
    data_list = []
    class_list = []

    max_bytes, min_bytes = -1, 999999
    size_dict = {}

    for class_nm in os.listdir(data_path):
        
        class_txt_cnt = 0
        class_flag = 0

        if class_nm == '.DS_Store': continue

        while class_flag != 1:
            for txt_file in os.listdir(data_path + class_nm):
                
                if txt_file.endswith('.txt'):
                    file_path = data_path + class_nm + '/' + txt_file
                
                    nonAsciiFile = {}

                    with open(file_path, 'r') as file:
                        class_label = class_nm

                        strings = file.readlines()
                        data = ' '.join(strings)

                        char_cnt = 0
                        data_124 = ''

                        for chars in data:
                            
                            if not chars.isascii():

                                if file_path not in nonAsciiFile:
                                    nonAsciiFile[file_path] = 1
                                else:
                                    nonAsciiFile[file_path] += 1
                                
                                info_recorder.record_nonAscii_txt(file_path + '\t : ' + chars)
                            
                            if chars.isascii():
                                
                                if char_cnt != 512:
                                    char_cnt += 1
                                    data_124 += chars
                                elif char_cnt == 512:
                                    char_cnt = 0
                                    class_txt_cnt += 1
                                    data_list.append(data_124)
                                    class_list.append(class_label)

                                    if not len(data_124) in size_dict:
                                        size_dict[len(data_124)] = 1
                                    else:
                                        size_dict[len(data_124)] += 1
                                    
                                    data_124 = ''

                                if class_txt_cnt == 10000:
                                    class_flag = 1
                                    break
                    
                    info_recorder.print_nonAscii_info(nonAsciiFile)
                
                if class_flag == 1:
                    break

        info_recorder.record_data_info(class_nm, class_txt_cnt)
    
    data = {'sourceCode': data_list, 'classLabel': class_list}
    df = pd.DataFrame(data)

    return df, [max_bytes, min_bytes], size_dict