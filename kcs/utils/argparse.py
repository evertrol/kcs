import argparse


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('-v', '--verbosity', action='count',
                    default=0, help="Verbosity level")
parser.add_argument('--nproc', type=int, default=1)
