import os

import numpy as np
import pandas as pd

from tone_util import tone_map


def calculate_total_weighted_durations_of_transitions(df: pd.DataFrame) -> float:
    return sum(df.apply(lambda row: (row['time1'] * row['time2']) / (row['pause'] / (row['time1'] * row['time2']) + 1),
                        axis=1))


def calculate_sum_of_durations_of_transitions(matrix) -> float:
    return sum(map(lambda row: sum(row), matrix))


def build_a_sum_of_duration_transitions_matrix_file(df: pd.DataFrame, file_name: str):
    matrix = np.empty((12, 12))
    for i in tone_map:
        for j in tone_map:
            transitions = df[(df['note1'] == tone_map[i]) & (df['note2'] == tone_map[j])]
            matrix[i][j] = calculate_total_weighted_durations_of_transitions(transitions)
    weighted_matrix = np.array(list(
        map(lambda row: list(map(lambda x: x / calculate_sum_of_durations_of_transitions(matrix), row)), matrix)))
    # Путь для сохранения файла
    file_path = f'{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}/weighted_matrix/{file_name}'  # Путь к файлу

    # Проверка наличия директории и создание, если отсутствует
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Сохранение массива в файл
    np.savetxt(file_path, weighted_matrix)
