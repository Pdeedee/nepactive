from nepactive import dlog,parse_yaml
from nepactive.train import Nepactive
from nepactive.remote import Remotetask
import logging
import argparse
import yaml


# 创建解析器



def _main():
    parser = argparse.ArgumentParser(description="nepactive")
    parser.add_argument("--remote", action="store_true", default=None, help="remote run")
    args = parser.parse_args()  # 解析命令行输入并获取参数
    idata:dict = parse_yaml("in.yaml")
    
    if args.remote:
        task = Remotetask(idata=idata)
        task.run_submission()
    else:
        task = Nepactive(idata=idata)
        task.run()
    
if __name__ == "__main__":
    _main()