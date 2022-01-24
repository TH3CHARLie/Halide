import sys

def transform(filename, outfile):
    with open(filename, "r") as f:
        with open(outfile, "w") as o:
            for line in f:
                newline = line.replace('.sample', '.newsample')
                o.write(newline)


if __name__ == "__main__":
    transform(sys.argv[1], sys.argv[2])