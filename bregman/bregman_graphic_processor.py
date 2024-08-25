import os

import gudhi
from matplotlib import pyplot as plt
from tqdm import tqdm

from lena.util.matrices_util import process_matrices

current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_directory)
matrix_directory = os.path.join(parent_directory, 'weighted_matrix')
for root, dirs, files in os.walk(matrix_directory):
    for dir in tqdm(dirs):
        stat, phs = process_matrices(f'{matrix_directory}/{dir}')
        arr = [[], []]
        arr[0] = list(map(lambda x: (0, tuple(x)), phs[0]))
        arr[1] = list(map(lambda x: (1, tuple(x)), phs[1]))

        gudhi.plot_persistence_barcode(arr[0])
        # plt.show()
        plt.savefig(f'{matrix_directory}/{dir}/bregman_barcode_0.png')
        gudhi.plot_persistence_barcode(arr[1])
        # plt.show()
        plt.savefig(f'{matrix_directory}/{dir}/bregman_barcode_1.png')
