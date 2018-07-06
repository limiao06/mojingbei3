import os
import sys
import time
import argparse


def main():
    parser = argparse.ArgumentParser(description='Read scan params results')
    # paths
    parser.add_argument("logfile", help="scan params log file")
    parser.add_argument("output", help="output")
    params, _ = parser.parse_known_args()

    input = open(params.logfile)
    line = input.readline()

    results = []
    while True:
        config = input.readline()
        if not line:
            break

        config_values = []
        tokens = config.split(",")
        for t in tokens:
            config_values.append(float(t.split(":")[1]))

        _ = input.readline()
        score = float(input.readline())
        config_values.append(score)
        results.append(config_values)


    df = pd.DataFrame(data=results, columns=["dpout_fc", "enc_lstm_dim", "fc_dim", "score"])
    df.to_csv(args.output)

if __name__ == '__main__':
    main()