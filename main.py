from nepactive import dlog,parse_yaml
from nepactive.train import Nepactive
from nepactive.remote import Remotetask
from nepactive.stable import StableRun
import logging
import argparse
import yaml


# 创建解析器



def main():
    parser = argparse.ArgumentParser(description="nepactive")
    parser.add_argument("--remote", action="store_true", default=None, help="remote run")
    parser.add_argument("--stable", action="store_true", default=None, help="remote run")
    parser.add_argument("--shock", action="store_true", default=None, help="shock velocity calculation")
    args = parser.parse_args()  # 解析命令行输入并获取参数
    idata:dict = parse_yaml("in.yaml")
    
    if args.remote:
        task = Remotetask(idata=idata)
        task.run_submission()
    elif args.stable:
        task = StableRun(idata=idata.get("stable"))
        task.run()
    elif args.shock:
        task = Nepactive(idata=idata)
        task.shock()
    else:
        task = Nepactive(idata=idata)
        task.run()
    
if __name__ == "__main__":
    main()