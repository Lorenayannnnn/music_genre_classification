
import os
from pathlib import Path

import pandas as pd
import torch
from pydub import AudioSegment

from AudioToGenreDataset import AudioToGenreDataset


def get_device(force_cpu, status=True):
    if not force_cpu and torch.cuda.is_available():
        device = torch.device("cuda")
        if status:
            print("Using CUDA")
    else:
        device = torch.device("cpu")
        if status:
            print("Using CPU")
    return device


def print_data_split_summary(split_name: str, split_data: list):
    print(split_name)
    genre_to_examples = {
        "blues": 0,
        "classical": 0,
        "country": 0,
        "disco": 0,
        "hiphop": 0,
        "jazz": 0,
        "metal": 0,
        "pop": 0,
        "reggae": 0,
        "rock": 0
    }
    for filename in split_data:
        genre = filename.split("/")[0]
        genre_to_examples[genre] += 1

    for genre, sample_num in genre_to_examples.items():
        print(genre, sample_num)


def load_split_dataframe(data_split_filename: str, data_dir: str):
    """
    Create dataframe for train, dev, test based on input data split file
    :param data_split_filename (../data_split.txt)
    :param data_dir (../data/genres_original/)

    :return: train_dataframe, dev_dataframe, test_dataframe
    """
    # Dataframe: {"audio_filename": [], "label": []}

    train_dataframe = {}
    dev_dataframe = {}
    test_dataframe = {}
    f = open(data_split_filename, "r")
    for i in range(3):
        split_name = f.readline().replace(":\n", "")
        split_data = f.readline().replace("['", "").replace("']", "").strip().split("', '")

        temp_audio_filenames = []
        temp_labels = []
        for filename in split_data:
            # filename: pop/pop00048.png
            filename_label_split = filename.split("/")      # Get genre
            genre = filename_label_split[0]
            filename_type_split = filename_label_split[1].split(".")    # For changing from png to wav
            file_id = filename_type_split[0].split(genre)[1]

            # Split audio into segments for data augmentation
            audio_segment_names = split_audio_file(
                data_dir=data_dir,
                genre=genre,
                file_id=file_id,
                original_filename=f"{genre}/{genre}.{file_id}.wav"
            )

            # w/o data augmentation
            # temp_audio_filenames.append(genre + '/' + genre + '.' + file_id + ".wav")
            # temp_labels.append(genre)

            # w/ data augmentation
            temp_audio_filenames.extend(audio_segment_names)
            temp_labels.extend([genre] * len(audio_segment_names))

        if split_name == "train":
            train_dataframe = pd.DataFrame({"audio_filename": temp_audio_filenames, "label": temp_labels})
        elif split_name == "dev":
            dev_dataframe = pd.DataFrame({"audio_filename": temp_audio_filenames, "label": temp_labels})
        else:
            test_dataframe = pd.DataFrame({"audio_filename": temp_audio_filenames, "label": temp_labels})

    f.close()

    return train_dataframe, dev_dataframe, test_dataframe


def create_dataset_w_dataframe(dataframe, root_dir: str, feature_extractor, normalize_audio_arr: bool, device):
    return AudioToGenreDataset(
        root_dir=root_dir,
        dataframe=dataframe,
        feature_extractor=feature_extractor,
        normalize_audio_arr=normalize_audio_arr,
        device=device
    )


def split_audio_file(data_dir: str, genre: str, file_id: str, original_filename: str):
    """
    Split input audio file into pieces of 3 seconds with 50%
    :param data_dir: ../data/genres_original/
    :param genre: pop
    :param file_id (00048)
    :param original_filename (pop/pop.00048.wav)

    Usage:
    audio_segments = split_audio_file(
        data_dir="../data/genres_original/",
        genre="pop",
        file_id="00048",
        original_filename="pop/pop.00048.wav"
    )

    :returns list of processed audio filenames (e.g.: split/pop/pop.00048.0.wav)
    """
    start = 0
    end = 3
    idx = 0

    # Read in input audio file
    original_audio = AudioSegment.from_wav(os.path.join(data_dir, original_filename))

    audio_segments = []
    # Each audio track is 30 seconds long
    while end < 30:
        segment = original_audio[(start * 1000):(end * 1000)]
        segment_filename = f"split/{genre}/{genre}.{file_id}.{idx}.wav"

        # Make dir if does not exist
        Path(os.path.join(data_dir, f"split/{genre}/")).mkdir(parents=True, exist_ok=True)

        # f = open(os.path.join(data_dir, segment_filename))
        segment.export(os.path.join(data_dir, segment_filename), format="wav")
        # f.close()

        audio_segments.append(segment_filename)
        start += 1.5
        end += 1.5
        idx += 1

    return audio_segments


def normalize_tensor(tensor_arr):
    mean, std = torch.mean(tensor_arr), torch.std(tensor_arr)
    normalized_arr = (tensor_arr - mean) / std
    return normalized_arr
