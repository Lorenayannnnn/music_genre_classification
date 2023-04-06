
import argparse
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import tqdm
from transformers import Wav2Vec2FeatureExtractor

from MusicGenreClassificationModel import MusicGenreClassificationModel, EnsembleMusicGenreClassificationModel
from AudioToGenreDataset import index_2_genre
from utils import get_device, load_split_dataframe, create_dataset_w_dataframe


def setup_dataloader(args, feature_extractor, device):
    # Create dataframe based on input data split
    print("Creating dataframes")
    train_dataframe, dev_dataframe, test_dataframe = load_split_dataframe(args.data_split_txt_filepath, args.data_dir)

    # Create dataset
    print("Creating datasets")
    train_dataset = create_dataset_w_dataframe(train_dataframe, args.data_dir, feature_extractor,
                                               args.normalize_audio_arr, device)
    dev_dataset = create_dataset_w_dataframe(dev_dataframe, args.data_dir, feature_extractor, args.normalize_audio_arr,
                                             device)
    test_dataset = create_dataset_w_dataframe(test_dataframe, args.data_dir, feature_extractor,
                                              args.normalize_audio_arr, device)

    # Create dataloader
    print("Setting up dataloader")
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
    dev_loader = DataLoader(dev_dataset, shuffle=True, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, shuffle=True, batch_size=args.batch_size)

    return train_loader, dev_loader, test_loader

def setup_criterion(args, device):
    criterion = torch.nn.CrossEntropyLoss().to(device)
    return criterion


def setup_optimizer(args, model):
    """
    return:
        - criterion: loss_fn
        - optimizer: torch.optim
    """
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.learning_rate)
    return optimizer


def setup_model(args, criterion):
    if ".ckpt" in args.model_name_or_path:
        # Load pretrained model
        print(f"Load pretrained model from {args.model_name_or_path}")
        model = torch.load(args.model_name_or_path)
    elif args.do_ensemble:
        biased_model = MusicGenreClassificationModel(
            # TODO change biased model
            model_name="facebook/wav2vec2-base",
            # model_name=args.model_name_or_path,
            freeze_part="none",
            process_last_hidden_state_method=args.process_last_hidden_state_method,
            freeze_layer_num=args.freeze_layer_num,
            criterion=criterion
        )
        main_model = MusicGenreClassificationModel(
            model_name=args.model_name_or_path,
            freeze_part=args.freeze_part,
            process_last_hidden_state_method=args.process_last_hidden_state_method,
            freeze_layer_num=args.freeze_layer_num,
            criterion=criterion
        )
        model = EnsembleMusicGenreClassificationModel(
            biased_model=biased_model,
            main_model=main_model,
            ensemble_ratio=args.ensemble_ratio
        )
    else:
        model = MusicGenreClassificationModel(
            model_name=args.model_name_or_path,
            freeze_part=args.freeze_part,
            process_last_hidden_state_method=args.process_last_hidden_state_method,
            freeze_layer_num=args.freeze_layer_num,
            criterion=criterion
        )
    return model


def train_epoch(
    args,
    model,
    loader,
    optimizer,
    criterion,
    device,
    training=True,
):
    if training:
        model.train()
    else:
        model.eval()
    epoch_loss = 0.0

    # keep track of the model predictions for computing accuracy
    pred_labels = []
    target_labels = []

    # iterate over each batch in the dataloader
    # NOTE: you may have additional outputs from the loader __getitem__, you can modify this
    for loaded_inputs in tqdm.tqdm(loader):
        # put model inputs to device
        inputs = loaded_inputs["input_values"][0]
        labels = loaded_inputs["label"]

        inputs, labels = inputs.to(device).float(), labels.to(device).long()

        # calculate the loss and train accuracy and perform backprop
        outputs = model(inputs, labels, do_train=training)
        loss = outputs.loss
        pred_logits = outputs.logits

        # step optimizer and compute gradients during training
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # logging
        epoch_loss += loss.item()

        # compute metrics
        preds = pred_logits.argmax(-1)
        pred_labels.extend(preds.cpu().numpy())
        target_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(pred_labels, target_labels)
    epoch_loss /= len(loader)

    # TODO
    # Reference: https://stackoverflow.com/questions/39770376/scikit-learn-get-accuracy-scores-for-each-class
    def report_classification_accuracy(pred_labels, true_labels):
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(pred_labels, true_labels)

        import numpy as np
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        acc_results = cm.diagonal()
        for idx, class_name in enumerate(index_2_genre):
            print(class_name, acc_results[idx])

    # TODO
    report_classification_accuracy(pred_labels=pred_labels, true_labels=target_labels)

    return epoch_loss, acc


def validate(args, model, loader, optimizer, criterion, device):
    # set model to eval mode
    model.eval()

    # don't compute gradients
    with torch.no_grad():
        val_loss, val_acc = train_epoch(
            args,
            model,
            loader,
            optimizer,
            criterion,
            device,
            training=False,
        )

    return val_loss, val_acc


def main(args):
    device = get_device(args.force_cpu)

    # Load feature extractor
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        args.feature_extractor_name if ".ckpt" in args.model_name_or_path else args.model_name_or_path)

    # get data loaders
    train_loader, dev_loader, test_loader = setup_dataloader(args, feature_extractor, device)
    loaders = {"train": train_loader, "val": dev_loader, "test": test_loader}

    criterion = setup_criterion(args, device)

    # build model
    model = setup_model(args, criterion=criterion).to(device)
    print(model)

    # get optimizer
    optimizer = setup_optimizer(args, model)

    all_train_acc = []
    all_train_loss = []
    all_val_acc = []
    all_val_loss = []

    best_val_acc = 0
    best_val_epoch = -1

    if args.do_train:
        for epoch in range(args.num_epochs):
            # train model for a single epoch
            print(f"Epoch {epoch}")
            train_loss, train_acc = train_epoch(
                args,
                model,
                loaders["train"],
                optimizer,
                criterion,
                device,
            )

            print(f"train loss : {train_loss} | train acc: {train_acc}")
            all_train_acc.append(train_acc)
            all_train_loss.append(train_loss)

            if epoch % args.val_every == 0:
                val_loss, val_acc = validate(
                    args,
                    model,
                    loaders["val"],
                    optimizer,
                    criterion,
                    device,
                )
                print(f"val loss : {val_loss} | val acc: {val_acc}")
                all_val_acc.append(val_acc)
                all_val_loss.append(val_loss)

                if val_acc > best_val_acc:
                    # Run test

                    test_loss, test_acc = validate(
                        args,
                        model,
                        loaders["test"],
                        optimizer,
                        criterion,
                        device,
                    )
                    print(f"test loss : {test_loss} | test acc: {test_acc}")

                    best_val_acc = val_acc
                    best_val_epoch = epoch
                    Path(args.outputs_edir).mkdir(parents=True, exist_ok=True)
                    ckpt_model_file = os.path.join(args.outputs_dir, "model.ckpt")
                    performance_file = os.path.join(args.outputs_dir, "results.txt")
                    print("saving model to ", ckpt_model_file)
                    torch.save(model, ckpt_model_file)
                    open(performance_file, 'w').write(
                        f"Train acc: {train_acc} | Dev acccuracy: {best_val_acc} | Test accuracy: {test_acc}")
    elif args.do_eval:
        # Load pretrained model
        model = torch.load(os.path.join(args.outputs_dir, "model.ckpt"))
        val_loss, val_acc = validate(
            args,
            model,
            loaders["val"],
            optimizer,
            criterion,
            device,
        )
        print(f"val loss : {val_loss} | val acc: {val_acc}")

    if args.do_test:
        if not args.do_train:
            # Load pretrained model
            model = torch.load(os.path.join(args.outputs_dir, "model.ckpt"))
        test_loss, test_acc = validate(
            args,
            model,
            loaders["test"],
            optimizer,
            criterion,
            device,
        )
        print(f"test loss : {test_loss} | test acc: {test_acc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--do_train", action="store_true", help="training mode")
    parser.add_argument("--do_eval", action="store_true", help="training mode")
    parser.add_argument("--do_test", action="store_true", help="training mode")

    parser.add_argument("--outputs_dir", type=str, default="output", help="where to save training outputs")
    parser.add_argument("--data_dir", type=str, help="where the music audio files are stored")
    parser.add_argument("--data_split_txt_filepath", type=str, default="../data_split.txt",
                        help="where the data split txt file is stored")

    parser.add_argument(
        "--batch_size", type=int, default=32, help="size of each batch in loader"
    )
    parser.add_argument("--force_cpu", action="store_true", help="debug mode")
    parser.add_argument(
        "--num_epochs", default=30, type=int, help="number of training epochs"
    )
    parser.add_argument(
        "--val_every",
        default=2,
        type=int,
        help="number of epochs between every eval loop",
    )

    parser.add_argument(
        "--model_name_or_path",
        default="facebook/wav2vec2-base",
        type=str,
        help="Path to local pretrained model or name of model from huggingface"
    )
    parser.add_argument(
        "--feature_extractor_name",
        default="facebook/wav2vec2-base",
        type=str,
        help="Name of feature extractor (load from huggingface)"
    )

    parser.add_argument(
        "--normalize_audio_arr",
        action='store_true',
        help="whether normalize audio array"
    )
    parser.add_argument(
        "--freeze_part",
        type=str,
        help="freeze which part of the loaded pretrained model: "
             "['full' (entire model), 'feature_extractor', 'none', 'freeze_encoder_layers']",
    )
    parser.add_argument(
        "--process_last_hidden_state_method",
        type=str,
        help="Method for processing last hidden state output from Wav2Vec2 model. Should choose from [last, average, sum, max]"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="learning rate of optimizer"
    )
    parser.add_argument(
        "--freeze_layer_num",
        type=int,
        default=12,
        help="Number of encoder layers (start counting from the low level) that will be freezed. Default: freeeze the entire model"
    )

    parser.add_argument(
        "--dropout_rate",
        type=float,
        default=0.1,
        help="dropout rate of dropout layer"
    )

    # Ensemble learning
    parser.add_argument(
        "--do_ensemble",
        action='store_true',
        help="Will create 2 models: 1 biased model without freezing any layers, and another main model created based on input arguments"
    )

    parser.add_argument(
        "--ensemble_ratio",
        type=float,
        default=0.1,
        help="percentage of biased prediction that will be included in the final result"
    )

    args = parser.parse_args()
    main(args)

