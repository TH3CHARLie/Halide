import os
import sys

def compute_valid_rate(output_dir):
    total = 0
    valid = 0
    for filename in os.listdir(output_dir):
        # count how many files have the suffix '.log
        if filename.endswith('.log'):
            total += 1
        if filename.endswith('.hlpipe'):
            valid += 1
    print('Total: %d, Valid: %d, Rate: %.2f%%' % (total, valid, valid * 100.0 / total))
    return valid * 100.0 / total


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: %s <output_dir>' % sys.argv[0])
        sys.exit(1)
    compute_valid_rate(sys.argv[1])
