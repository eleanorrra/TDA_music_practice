import os

import mido
from tqdm import tqdm


def filter_events(midi_file):
    non_meta_tracks = midi_file.tracks[1:]
    events = {}
    for i in range(len(non_meta_tracks)):
        events[i] = [event for event in non_meta_tracks[i] if
                     not event.is_meta and (event.type == 'note_on' or event.type == 'note_off')]
    return events


def parse_midis(path='midis'):
    midis = {}
    current_directory = os.path.dirname(os.path.abspath(__file__))
    parent_directory = os.path.dirname(current_directory)
    midis_directory = os.path.join(parent_directory, path)
    for root, dirs, files in os.walk(midis_directory):
        for file in tqdm(files):
            file_directory = os.path.join(root, file)
            midi_file = mido.MidiFile(file_directory, clip=True)
            events = filter_events(midi_file)
            midis[file] = events
    return midis
