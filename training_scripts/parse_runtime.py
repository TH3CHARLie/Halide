#!/usr/bin/env python
import os
import sys
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, action="store", help="path to prediction with filename")
    parser.add_argument('--output', required=True, action="store", help="path to csv output")
    args = parser.parse_args()
    return args


def parse_runtime(infile):
    lines = []
    with open(infile, "r") as f:
        lines = f.readlines()
    lines = [l.strip() for l in lines]
    runtimes = []
    lines = lines[2:]
    for line in lines[4:]:
        tokens = line.split()
        stage_name = tokens[1][:-1]
        stage_runtime = float(tokens[2][:-2])
        runtimes.append((stage_name, stage_runtime))
    return runtimes


def write_runtimes(runtimes, outfile):
    with open(outfile, "w") as f:
        for runtime in runtimes:
            f.write(f'{runtime[0]} {runtime[1]}\n')


def main():
    args = parse_args()
    runtimes = parse_runtime(args.input)
    write_runtimes(runtimes, args.output)


if __name__ == "__main__":
    main()