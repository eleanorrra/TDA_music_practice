import mido
import pandas as pd
from tqdm import tqdm

from tone_util import to_tone


def to_series(note_off_1: mido.Message, note_on: mido.Message, note_off_2: mido.Message) -> pd.Series:
    return pd.Series({
        "note1": to_tone(note_off_1.note),
        "note2": to_tone(note_off_2.note),
        "time1": note_off_1.time,
        "time2": note_off_2.time,
        "pause": note_on.time
    })


def set_double_chord(filtered_events, i, df):
    df = pd.concat([df, to_series(filtered_events[i - 1], filtered_events[i], filtered_events[i + 2]).to_frame().T],
                   ignore_index=True)
    df = pd.concat([df, to_series(filtered_events[i - 1], filtered_events[i + 1], filtered_events[i + 3]).to_frame().T],
                   ignore_index=True)

    df = pd.concat([df, to_series(filtered_events[i + 2], filtered_events[i + 4], filtered_events[i + 5]).to_frame().T],
                   ignore_index=True)
    return pd.concat(
        [df, to_series(filtered_events[i + 3], filtered_events[i + 4], filtered_events[i + 5]).to_frame().T],
        ignore_index=True)


def set_triple_chord(filtered_events, i, df):
    df = pd.concat([df, to_series(filtered_events[i - 1], filtered_events[i], filtered_events[i + 3]).to_frame().T],
                   ignore_index=True)
    df = pd.concat([df, to_series(filtered_events[i - 1], filtered_events[i + 1], filtered_events[i + 4]).to_frame().T],
                   ignore_index=True)
    df = pd.concat([df, to_series(filtered_events[i - 1], filtered_events[i + 2], filtered_events[i + 5]).to_frame().T],
                   ignore_index=True)

    df = pd.concat([df, to_series(filtered_events[i + 3], filtered_events[i + 6], filtered_events[i + 7]).to_frame().T],
                   ignore_index=True)
    df = pd.concat([df, to_series(filtered_events[i + 4], filtered_events[i + 6], filtered_events[i + 7]).to_frame().T],
                   ignore_index=True)
    return pd.concat(
        [df, to_series(filtered_events[i + 5], filtered_events[i + 6], filtered_events[i + 7]).to_frame().T],
        ignore_index=True)


def to_df(events: [mido.Message]):
    df = pd.DataFrame(columns=["note1", "note2", "time1", "time2", "pause"])
    for i in range(2, len(events) - 7):
        if events[i].type == 'note_on' and events[i + 1].type == 'note_off':
            df = pd.concat([df, to_series(events[i - 1], events[i], events[i + 1]).to_frame().T],
                           ignore_index=True)
        elif events[i].type == 'note_on' and events[i + 1].type == 'note_on' and events[
            i + 2].type != 'note_on' and events[i + 4].type == 'note_on':
            df = set_double_chord(events, i, df)
        elif events[i].type == 'note_on' and events[i + 1].type == 'note_on' and events[
            i + 2].type == 'note_on':
            set_triple_chord(events, i, df)
    df.drop(df[df['time1'] == 0].index, inplace=True)
    df.drop(df[df['time2'] == 0].index, inplace=True)
    return df


def map_to_df(midis):
    mapped_data = {}
    for file_name, tracks in tqdm(midis.items()):
        mapped_data[file_name] = {}
        for track_num, events in tracks.items():
            mapped_data[file_name][track_num] = to_df(events)
    return mapped_data
