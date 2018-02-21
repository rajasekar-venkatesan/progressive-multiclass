# Imports
import argparse
from plt_mcc import plt_multiclass
from log_utils import Logger


# Functions
def parse_command_line_args():
    parser = argparse.ArgumentParser(description='Progressive Multiclass ELM')
    parser.add_argument("-f", "--filename", default="../datasets/iris_plt.csv", help="Filename of the dataset (csv file)")
    parser.add_argument("-hr", "--header", default="infer", help="Header arguement for read_csv in pandas")
    parser.add_argument("-l", "--label", default="last", help="Location of label column in the csv file")
    parser.add_argument("-s", "--scale", default="minmax", help="Scaling type for feature scaling")
    parser.add_argument("-t", "--testratio", default=0.1, type=float, help="Ratio of test samples to total samples")
    parser.add_argument("-n", "--neurons", default=10, type=int, help="Number of neurons in hidden layer")
    parser.add_argument("-i", "--initial", default=30, type=int, help="Number of samples in initial block")
    parser.add_argument("-b", "--batch", default=1, type=int, help="Batch size for sequential training")
    args = parser.parse_args()
    return args


def main(args):
    logger_config = Logger(__name__, log_file='plt_multiclass', log_level='debug', show_log=False)
    logger = logger_config.get_logger()
    try:
        logger.info('Executing plt_multiclass with the arguements received')
        plt_multiclass(logger, **vars(args))
    except:
        print(f'\n\n!!! Error Occurred During Execution !!!\nCheck log file for further details\n'
              f'Log file: {logger.handlers[0].baseFilename}')
    pass


# Main
if __name__ == '__main__':
    args = parse_command_line_args()
    main(args)