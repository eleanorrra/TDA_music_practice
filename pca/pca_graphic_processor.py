import os
import re

import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from functools import reduce

from tqdm import tqdm

from lena.util.matrices_util import process_matrices

current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_directory)
matrix_directory = os.path.join(parent_directory, 'weighted_matrix')
composers_genres_stats = {}
for root, dirs, files in os.walk(matrix_directory):
    for dir in tqdm(dirs):
        stat, _ = process_matrices(f'{matrix_directory}/{dir}')
        if not (dir.split('_')[0] in composers_genres_stats):
            composers_genres_stats[dir.split('_')[0]] = {}
        composers_genres_stats[dir.split('_')[0]].update({dir.split('_')[1] + dir.split('_')[2].split('.')[0]: stat})

composers = list(composers_genres_stats.keys())
genres = reduce(lambda keys, d: keys | set(d.keys()), composers_genres_stats.values(), set())


def StatPCA(graphic_genre=''):
    genre_stats = {value: [] for value in composers}
    if graphic_genre != '':
        for composer in genre_stats.keys():
            composer_genres = composers_genres_stats[composer]
            filtered_genres = {key: value for key, value in composer_genres.items() if key.startswith(graphic_genre)}
            genre_stats[composer] = list(filtered_genres.values())
    else:
        for composer in genre_stats.keys():
            composer_genres = composers_genres_stats[composer]
            genre_stats[composer] = list(composer_genres.values())
    genre_beethoven = np.array(genre_stats['Beethoven'])
    genre_mozart = np.array(genre_stats['Mozart'])
    genre_haydn = np.array(genre_stats['Haydn'])

    gen_all = np.concatenate([genre_beethoven, genre_mozart, genre_haydn], axis=0)
    pca = PCA(2)
    pca.fit(gen_all)

    pca_beethoven = pca.transform(genre_beethoven)
    pca_mozart = pca.transform(genre_mozart)
    pca_haydn = pca.transform(genre_haydn)

    plt.scatter(pca_beethoven[:, 0], pca_beethoven[:, 1], c='blue', label='Beethoven', linewidth=0.1, alpha=0.5)
    plt.scatter(pca_mozart[:, 0], pca_mozart[:, 1], c='red', label='Mozart', linewidth=0.1, alpha=0.5)
    plt.scatter(pca_haydn[:, 0], pca_haydn[:, 1], c='green', label='Haydn', linewidth=0.1, alpha=0.5)

    plt.legend()
    plt.xlabel('1st principal component')
    plt.ylabel('2nd principal component')
    plt.title(f'{graphic_genre} PCA')
    plt.show()

    plt.bar(composers, [np.std(pca_beethoven), np.std(pca_haydn), np.std(pca_mozart)])
    plt.title('Standard deviation of PCA decomposotion')
    plt.show()


def StatPCAGenre(graphic_composer='Beethoven'):
    genres_by_composer = composers_genres_stats[graphic_composer]
    genres_by_composer_map = {}
    for genre in genres_by_composer:
        genre_name = re.split(r'(\d)', genre, 1)[0]
        if not (genre_name in genres_by_composer_map):
            genres_by_composer_map[genre_name] = []
        genres_by_composer_map[genre_name].append(genres_by_composer[genre])
    colors = ['blue', 'red', 'green', 'black', 'orange']

    for genre, genre_statistic in genres_by_composer_map.items():
        genres_by_composer_map[genre] = np.array(genre_statistic)

    genre_map = {genre: np.vstack(substats) for genre, substats in genres_by_composer_map.items()}

    genre_all = np.array([stats for substats in genres_by_composer_map.values() for stats in substats])
    pca_composer = PCA(2)
    pca_composer.fit(genre_all)

    pca_arr = list(map(pca_composer.transform, genre_map.values()))
    for i in range(len(genres_by_composer_map.keys())):
        plt.scatter(pca_arr[i][:, 0], pca_arr[i][:, 1], c=colors[i], label=list(genres_by_composer_map.keys())[i], linewidth=0.1, alpha=0.5)

    plt.legend()
    plt.xlabel('1st principal component')
    plt.ylabel('2nd principal component')
    plt.title(f'{graphic_composer} PCA')
    plt.show()

    plt.bar(list(genres_by_composer_map.keys()), list(map(np.std, pca_arr)))
    plt.title('Standard deviation of PCA decomposotion')
    plt.show()


StatPCA()

StatPCA('Adagio')
StatPCA('Allegro')
StatPCA('Andante')

StatPCAGenre(graphic_composer='Haydn')
StatPCAGenre(graphic_composer='Mozart')
StatPCAGenre(graphic_composer='Beethoven')
