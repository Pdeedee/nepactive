from nepactive import dlog
from nepactive.train import run
import logging

def _main():

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    # logging.basicConfig (filename="compute_string.log", filemode="a", level=logging.INFO, format='%(asctime)s %(message)s')
    logging.getLogger("paramiko").setLevel(logging.WARNING)

    logging.info("start running")
    # run_iter(args.PARAM, args.MACHINE)
    logging.info("finished!")
    run()


if __name__ == "__main__":
    _main()