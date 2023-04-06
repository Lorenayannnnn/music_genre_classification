
import torch
import torchaudio
from torch.utils.data.dataset import Dataset

genre_2_index = {
    "blues": 0,
    "classical": 1,
    "country": 2,
    "disco": 3,
    "hiphop": 4,
    "jazz": 5,
    "metal": 6,
    "pop": 7,
    "reggae": 8,
    "rock": 9
}
index_2_genre = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]

class AudioToGenreDataset(Dataset):
    def __init__(self, root_dir, dataframe, feature_extractor, normalize_audio_arr, device):
        # dataframe = {"audio_filename": [], "label": []}
        self.data_dict = dataframe
        self.root_dir = root_dir
        # Hubert feature extractor
        self.feature_extractor = feature_extractor

        # Whether Normalize audio arr
        self.normalize_audio_arr = normalize_audio_arr

        self.device = device

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # root: data/genres_original/
        audio_file_name = self.root_dir + self.data_dict.loc[idx, "audio_filename"]
        label = genre_2_index[self.data_dict.loc[idx, "label"]]

        # Convert audio to array of float
        processed_audio_data = convert_audio_to_float_input_values(audio_file_name,
                                                                   self.feature_extractor.sampling_rate, self.device)

        inputs = self.feature_extractor(processed_audio_data, sampling_rate=self.feature_extractor.sampling_rate,
                                        max_length=self.feature_extractor.sampling_rate, truncation=True)

        inputs["label"] = torch.tensor(label, dtype=torch.long)
        return inputs


def convert_audio_to_float_input_values(audio_file_name: str, extractor_sample_rate: int, device="cpu"):
    """
    Read input audio file and convert the file into the required sample rate
    :return audio: array of float
    """
    audio_array, sample_rate = torchaudio.load(audio_file_name)
    audio_array = audio_array.to(device)
    audio_array = torchaudio.functional.resample(audio_array, sample_rate, extractor_sample_rate).squeeze()

    return audio_array
