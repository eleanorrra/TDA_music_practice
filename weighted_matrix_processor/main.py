from tqdm import tqdm

from df_mapper import map_to_df
from midi_parser import parse_midis
from weighted_matrix_file_builder import build_a_sum_of_duration_transitions_matrix_file

midis = parse_midis()

notes_df = map_to_df(midis)

for file, tracks in notes_df.items():
    for track, df in tqdm(tracks.items()):
        build_a_sum_of_duration_transitions_matrix_file(df, f'{file}/{track}.txt')
