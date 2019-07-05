import pathlib
import subprocess


DATA_FILE = '/home/ubuntu/repos/pychess/data.csv'


def remove_duplicate_lines():
    new_output = subprocess.check_output(['sort', '-u', DATA_FILE])
    new_filename = DATA_FILE.replace('.csv', '_ready.csv')
    with open(new_filename, 'wb') as f:
        f.write(new_output)


def main():
    remove_duplicate_lines()
    file = pathlib.Path(DATA_FILE)
    file.unlink()


if __name__ == '__main__':
    main()
