"""
This script reads the pickle file of books and creates the txt files for training.
"""
import os
import pickle
from pathlib import Path


def main():
    """
    Main function that reads all the pickle files and creates text files.
    """
    dirpath: str = os.path.join('/Users/siddharthsaha/Downloads/1-500')
    for filename in os.listdir(dirpath):
        filepath = os.path.join(dirpath, filename)

        book = None
        with open(filepath, 'rb') as f:
            book = pickle.load(f)

        # write book to txt file
        target_file = os.path.join(
            os.path.join(Path(__file__).resolve().parents[2], 'data'),
            f'{os.path.splitext(filename)[0]}.txt',
        )
        with open(target_file, 'w', encoding='utf-8') as f:
            f.write(book)
        print(f'Wrote to file: {target_file}')


if __name__ == '__main__':
    main()
