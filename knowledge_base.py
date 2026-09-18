import os
import config_data as config

def check_md5(md5_str:str):
    if not os.path.exists(config.md5_path):
        open(config.md5_path,'w',encoding='utf-8').close()
        return False
    else:
        for line in open(config.md5_path,'r',encoding='utf-8').readlines():
            line=line.strip()
            
    

def save_md5():
    pass

def get_string_md5():
    pass


class knowlegdeBaseService(object):
    def __init__(self):
        self.chroma=None
        self.splitter=None

    def upload_by_str(self,data,filename):
        pass