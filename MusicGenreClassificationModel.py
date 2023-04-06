
import torch
from torch import nn
from transformers import Wav2Vec2Model, Wav2Vec2Config
from torch.nn import functional as F
from transformers.modeling_outputs import SequenceClassifierOutput


class MusicGenreClassificationModel(nn.Module):

    def __init__(self, model_name, freeze_part: str, process_last_hidden_state_method: str, criterion, dropout_rate=0.1,
                 freeze_layer_num=12):
        super(MusicGenreClassificationModel, self).__init__()
        self.config = Wav2Vec2Config.from_pretrained(model_name)
        self.wav2vec2_model = Wav2Vec2Model.from_pretrained(model_name)
        self.process_last_hidden_state_method = process_last_hidden_state_method  # average, sum, max

        self.init_freeze(freeze_part, freeze_layer_num)

        self.criterion = criterion

        # Classifier head
        # self.projector = nn.Linear(in_features=self.config.hidden_size, out_features=int(self.config.hidden_size / 2))
        # self.classifier = nn.Linear(in_features=int(self.config.hidden_size / 2), out_features=10)

        self.projector = nn.Linear(in_features=self.config.hidden_size, out_features=self.config.hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.tanh_f = nn.Tanh()
        self.classifier = nn.Linear(in_features=self.config.hidden_size, out_features=10)

    def init_freeze(self, freeze_part: str, freeze_layer_num: int):
        """
        :param freeze_part: ['full' (entire model), 'feature_extractor', 'none', 'freeze_encoder_layers']
        """
        if freeze_part is None:
            raise ValueError(
                "Need to select part of the pretrained model that will be freeze but got None. Choose from the following choices: full, feature_extractor, none")
        if freeze_part == "full":
            self.freeze_entire_model()
        elif freeze_part == "feature_extractor":
            self.wav2vec2_model.feature_extractor._freeze_parameters()
        elif freeze_part == "freeze_encoder_layers":
            self.freeze_entire_model()
            for i, layer in reversed(list(enumerate(self.wav2vec2_model.encoder.layers))):
                if i < freeze_layer_num:
                    break
                for param in layer.parameters():
                    param.requires_grad = True

    def freeze_entire_model(self):
        for p in self.wav2vec2_model.parameters():
            p.requires_grad = False

    def process_last_hidden_state(self, hidden_states):
        # last, average, sum, max
        if self.process_last_hidden_state_method == "last":
            return hidden_states[:, -1, :]
        elif self.process_last_hidden_state_method == "average":
            return torch.mean(hidden_states, dim=1)
        elif self.process_last_hidden_state_method == "sum":
            return torch.sum(hidden_states, dim=1)
        elif self.process_last_hidden_state_method == "max":
            return torch.max(hidden_states, dim=1)[0]
        else:
            raise ValueError(
                f"Should pick one of the following method: last, average, sum, max (but got {self.process_last_hidden_state_method})")

    def forward(self, inputs, labels, do_train=True):
        # Freeze the model (use as feature extractor)
        hidden_states = self.wav2vec2_model(inputs).last_hidden_state
        processed_hidden_states = self.process_last_hidden_state(hidden_states)

        # Classifier head
        projector_outputs = self.projector(processed_hidden_states)
        tanh_outputs = self.tanh_f(projector_outputs)
        dropout_outputs = self.dropout(tanh_outputs)
        logits = self.classifier(dropout_outputs)

        # return outputs
        loss = None
        if labels is not None:
            loss = self.criterion(logits.squeeze(), labels)

        return SequenceClassifierOutput(
            logits=logits,
            loss=loss
        )


class EnsembleMusicGenreClassificationModel(nn.Module):
    """
    Ensemble biased & main model
    """

    def __init__(self, biased_model: MusicGenreClassificationModel, main_model: MusicGenreClassificationModel,
                 ensemble_ratio):
        super(EnsembleMusicGenreClassificationModel, self).__init__()

        self.biased_model = biased_model
        self.main_model = main_model
        self.ensemble_ratio = ensemble_ratio

    def forward(self, inputs, labels, do_train=True):
        main_logits = self.main_model(inputs, labels).logits

        with torch.no_grad():
            biased_logits = self.biased_model(inputs, labels).logits

        loss = None

        if labels is not None:
            loss = self.bias_add_loss(
                main_logits=main_logits,
                biased_logits=biased_logits,
                labels=labels,
                ensemble_ratio=self.ensemble_ratio
            )

        # During inference time, only use predictions from the main model
        if not do_train:
            return SequenceClassifierOutput(
                loss=loss,
                logits=main_logits
            )

        # Product of expert
        ensemble_logits = F.softmax(biased_logits, -1) * F.softmax(main_logits, -1)

        return SequenceClassifierOutput(
            loss=loss,
            logits=ensemble_logits
        )

    def bias_add_loss(self, main_logits, biased_logits, labels, ensemble_ratio):
        main_probs = F.softmax(main_logits, 1)
        bias_probs = F.softmax(biased_logits, 1)
        ensemble_probs = (1 - ensemble_ratio) * main_probs + ensemble_ratio * bias_probs
        return F.nll_loss(torch.log(ensemble_probs), labels).float()
