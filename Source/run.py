import argparse
from .DTW import DTW

parser = argparse.ArgumentParser(description='Configuration')
parser.add_argument('-gt', '--groundtruth', type = str, help = 'Path to database', required=True, metavar='')
parser.add_argument('-p', '--predict', type = str, help = 'Path to predict', required=True, metavar='')
parser.add_argument('-t', '--type', type = str, help = 'Should be in [1v1, series]', default='1v1', metavar='')
parser.add_argument('-r', '--reduction', type = str, help = 'If mean return mean of the series else None', default='mean', metavar='')

args = parser.parse_args()

def main():
  dtw = DTW()
  

if __name__ == '__main__':
  print('Ok')
  print('test')
  
  
