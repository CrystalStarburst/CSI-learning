import numpy as np
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("CM_path", type = Path, help="path to confusion matrix file")
    args = parser.parse_args()

    matrix = np.load(args.CM_path)
    print(matrix)

if __name__ == "__main__":
    main()