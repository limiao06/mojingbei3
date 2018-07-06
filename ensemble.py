# refine model preds with sim set outputs
import sys
import pandas as pd
import numpy as np
import argparse
import os
from inference import make_submission


def read_result(file):
    result = pd.read_csv(file)
    return result['y_pre'].values

def main():
    parser = argparse.ArgumentParser(description='ensemble')
    # paths
    parser.add_argument("model_outputs_dir", help="dir that contains all models output")
    parser.add_argument("output", help="output")

    params, _ = parser.parse_known_args()

    # print parameters passed, and all parameters
    print('\ntogrep : {0}\n'.format(sys.argv[1:]))
    print(params)

    results = []

    for f in os.listdir(params.model_outputs_dir):
        results.append(read_result(os.path.join(params.model_outputs_dir, f)))

    results = np.mean(results, axis=0)
    make_submission(results, params.output)

if __name__ == '__main__':
    main()
