from accelerate.data_loader import prepare_data_loader
from transformers import GPT2LMHeadModel, GPT2Config
from accelerate import notebook_launcher
from torch.utils.data import DataLoader
from accelerate import Accelerator
import torch.optim as optim
import torch.utils.data
from glob import glob
import argparse
import logging
import pickle
# import psutil
import torch
import json
import time
import math
import sys
import os

# https://docs.python.org/3/library/logging.html

# Setup logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

# https://pytorch.org/tutorials/beginner/transformer_tutorial.html


def load_pkl_file(data_dir):
    PATH = os.path.join(data_dir)
    EXT = "*.pkl"

    pkl_files = []
    for path, subdir, files in os.walk(PATH):
        for file in glob(os.path.join(path, EXT)):
            pkl_files.append(file)

    print("pkl files:", len(pkl_files))

    return pkl_files


def initialize_model(lr, momentum):
    # https://huggingface.co/docs/transformers/model_doc/gpt2#transformers.GPT2Config.n_embd
    with open("vocab/vocab.txt", "r") as fh:
        VOCAB_SIZE = len(fh.read().split("\n"))
    config = {
        "vocab_size": VOCAB_SIZE,  # Defines the number of different tokens that can be represented by the inputs_ids passed when calling GPT2Model or TFGPT2Model.
        "n_embd": 512,  # Dimensionality of the embeddings and hidden states.
        "n_layer": 4,  # Number of hidden layers in the Transformer encoder.
        "n_head": 8,  # Number of attention heads for each attention layer in the Transformer encoder.
        "n_inner": 2048,  # Dimensionality of the inner feed-forward layers.
        "use_cache": False,  # TODO: !!
    }
    configuration = GPT2Config(**config)
    model = GPT2LMHeadModel(configuration)
    model.gradient_checkpointing_enable()
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    return model, optimizer


# https://huggingface.co/transformers/v3.5.1/custom_datasets.html
# https://pytorch.org/docs/stable/data.html
# https://pytorch.org/docs/stable/data.html#map-style-datasets
# https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset
# https://yizhepku.github.io/2020/12/26/dataloader.html
class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: val[idx].detach().clone() for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)


def save_model(model, model_dir, accelerator):
    path = os.path.join(model_dir, "model.pth")
    model = accelerator.unwrap_model(model)
    torch.save(model.cpu().state_dict(), path)


def train(args):
    # Init
    use_cuda = args.num_gpus > 0
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(args.random_seed)
    if use_cuda:
        torch.cuda.manual_seed(args.random_seed)

    # Loading pkl files
    pkl_files = load_pkl_file(args.data_dir)
    pkl_files = pkl_files[:200]  # Limit to 200 files for testing

    # # Retrieving information on running processes and system utilization (CPU, memory, disks, network, sensors)
    # # https://pypi.org/project/psutil/
    # process = psutil.Process(os.getpid())
    # print("Memory Info:", (process.memory_info().rss))  # in bytes

    model, optimizer = initialize_model(
        args.lr,
        args.momentum,
    )
    # https://huggingface.co/docs/accelerate/accelerator
    accelerator = Accelerator(
        dispatch_batches=False,  # If set to True, the dataloader prepared by the Accelerator is only iterated through on the main process and then the batches are split and broadcast to each process. Will default to True for DataLoader whose underlying dataset is an IterableDataset, False otherwise.
        split_batches=False,  # Whether or not the accelerator should split the batches yielded by the dataloaders across the devices.
        fp16=True,  # Mixed precision training.
    )
    model, optimizer = accelerator.prepare(model, optimizer)

    print("Accelerator Info")
    print(accelerator.device)
    print(accelerator.state)

    device_idx = int(str(accelerator.state).split("\n")[2].split(":")[-1])  # TODO: !!
    offset = math.ceil(
        len(pkl_files)
        / int(str(accelerator.state).split("\n")[1].split(":")[-1])  # TODO: !!
    )
    start_idx = device_idx * offset

    for epoch in range(args.epochs):
        print(f"Epoch: {epoch}")
        print(f"from {start_idx} to {start_idx + offset}")

        for pkl_idx, pkl_file in enumerate(pkl_files[start_idx : start_idx + offset]):
            try:
                start_time = time.time()

                with open(pkl_file, "rb") as fh:
                    inputs = pickle.load(fh)

                # Create dataset from tokenization
                dataset = SequenceDataset(inputs)

                # https://pytorch.org/docs/stable/data.html
                loader = DataLoader(
                    dataset,
                    batch_size=args.batch_size,
                    shuffle=True,
                    num_workers=4,
                    pin_memory=True,  # Automatically put the fetched data Tensors in pinned memory, and thus enables faster data transfer to CUDA-enabled GPUs.
                )

                # https://huggingface.co/docs/accelerate/main/en/internal#accelerate.data_loader.prepare_data_loader
                # Wraps a PyTorch DataLoader to generate batches for one of the processes only.
                loader = prepare_data_loader(
                    loader,
                    split_batches=False,
                    put_on_device=True,
                    dispatch_batches=True,
                )
                end_time = time.time()
                print(
                    f"Prepared pkl #{pkl_idx+1} of device #{device_idx} in {(end_time - start_time)} seconds"
                )
            except Exception as e:
                print(f"CORRUPTED FILE: {pkl_file}")
                print(f"EXCEPTION: {e}")
                continue

            for batch_idx, batch in enumerate(loader):
                start_time = time.time()

                print(next(model.parameters()))
                outputs = model(**batch)
                print(outputs.loss, outputs.logits.shape)

                accelerator.backward(outputs.loss)  # TODO: !!
                optimizer.step()
                for param in model.parameters():
                    param.grad.zero_()
                end_time = time.time()
                print(
                    f"Trained batch #{batch_idx} of device #{device_idx} in {(end_time - start_time)} seconds"
                )

            if pkl_idx % 2 == 0 and device_idx == 0:
                print(f"Saving model in pkl_idx #{pkl_idx} of device_idx #{device_idx}")
                accelerator.wait_for_everyone()
                save_model(model, args.model_dir, accelerator)
        # https://huggingface.co/docs/accelerate/accelerator
        accelerator.wait_for_everyone()  # to make sure all processes join that point before continuing (useful before a model save for instance).
        save_model(model, args.model_dir, accelerator)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        metavar="BS",
        help="input batch size for training (default: 128)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-5,
        metavar="LR",
        help="learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        metavar="M",
        help="SGD momentum (default: 0.5)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        metavar="RS",
        help="random seed (default: 42)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        help="backend for distributed training (tcp, gloo on cpu and gloo, nccl on gpu)",
    )

    # Container environment
    parser.add_argument(
        "--hosts",
        type=list,
        default=json.loads(os.environ["SM_HOSTS"]),
    )
    parser.add_argument(
        "--current-host",
        type=str,
        default=os.environ["SM_CURRENT_HOST"],
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=os.environ["SM_MODEL_DIR"],
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.environ["SM_OUTPUT_DIR"],
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=os.environ["SM_CHANNEL_TRAINING"],
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=os.environ["SM_NUM_GPUS"],
    )

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(
        ",".join([str(npgu) for npgu in range(args.num_gpus)])
    )

    print(f"Instance with {args.num_gpus} GPUs in total")

    notebook_launcher(train, (args,), num_processes=args.num_gpus)
