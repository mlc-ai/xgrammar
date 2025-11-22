import csv
import sys


def replace_from_csv(csv_file, input_file):
    with open(input_file, "r", encoding="utf-8") as f:
        content = f.read()

    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        for old, _, new in reader:
            content = content.replace(old, new)

        with open(input_file, "w", encoding="utf-8") as f:
            f.write(content)


if __name__ == "__main__":

    csv_file = sys.argv[1]
    input_file = sys.argv[2]

    replace_from_csv(csv_file, input_file)
