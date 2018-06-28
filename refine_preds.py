# refine model preds with sim set outputs
import sys
import pandas as pd
import argparse
from inference import make_submission

def main():
    parser = argparse.ArgumentParser(description='refine model preds with sim set outputs')
    # paths
    parser.add_argument("--model_output", type=str, default='', help="model preds")
    parser.add_argument("--simset_output", type=str, default='output/sim_set_output.csv', help="sim set output")
    parser.add_argument("--output", type=str, default='output')

    params, _ = parser.parse_known_args()

    # print parameters passed, and all parameters
    print('\ntogrep : {0}\n'.format(sys.argv[1:]))
    print(params)

    model_output = pd.read_csv(params.model_output)
    simset_output = pd.read_csv(params.simset_output)

    results = []
    for m,s in zip(list(model_output['y_pre'].values), list(simset_output['y_pre'].values)):
        if s < 0:
            results.append(m)
        else:
            results.append(s)

    make_submission(results, params.output)

if __name__ == '__main__':
    main()
