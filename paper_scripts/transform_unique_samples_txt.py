import sys

def transform(filename, outfile):
    lines = []
    with open(filename, "r") as f:
        for line in f:
            newline = line.replace('.sample', '.newsample')
            lines.append(newline)
    with open(outfile, "w") as o:
        for newline in lines:
                o.write(newline)


if __name__ == "__main__":
    transform(sys.argv[1], sys.argv[2])