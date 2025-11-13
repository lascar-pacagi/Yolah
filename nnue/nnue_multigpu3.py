import numpy as np
import torch
from torch import nn
from torch.nn.functional import relu, softmax
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os
from tqdm import tqdm
import re
from torch.utils.data import Dataset, DataLoader, random_split
import glob
import sys
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
sys.path.append("../server")
from yolah import Yolah, Move, Square
import itertools

torch.set_float32_matmul_precision('high')

def preprocess_dataset_to_memmap(games_dir, output_prefix):
    """
    Convert GameDataset to memory-mapped files for efficient multi-worker loading.
    Run this once to preprocess your data.

    Args:
        games_dir: Directory containing game files
        output_prefix: Path prefix for output memmap files (e.g., "data/dataset")
    """
    print(f"Loading dataset from {games_dir}...")
    dataset = GameDataset(games_dir)
    total = len(dataset)

    print(f"Creating memory-mapped arrays for {total} samples...")

    # Create memory-mapped arrays
    inputs_mmap = np.memmap(f'{output_prefix}_inputs.npy', dtype=np.uint8,
                            mode='w+', shape=(total, INPUT_SIZE))
    outputs_mmap = np.memmap(f'{output_prefix}_outputs.npy', dtype=np.uint8,
                             mode='w+', shape=(total,))

    # Fill memory-mapped arrays
    for i in tqdm(range(total), desc="Preprocessing dataset"):
        inputs_mmap[i] = dataset.inputs[i]
        outputs_mmap[i] = dataset.outputs[i]

    # Flush to disk
    inputs_mmap.flush()
    outputs_mmap.flush()

    print(f"Dataset preprocessing complete!")
    print(f"  Inputs:  {inputs_mmap.nbytes / (1024**2):.2f} MB")
    print(f"  Outputs: {outputs_mmap.nbytes / (1024**2):.2f} MB")
    print(f"  Total:   {(inputs_mmap.nbytes + outputs_mmap.nbytes) / (1024**2):.2f} MB")

def bit_not(n, numbits=64):
    return (1 << numbits) - 1 - n

def bitboard64_to_list(n):
    b = [int(digit) for digit in bin(n)[2:]]
    return [0]*(64 - len(b)) + b

class GameDataset(Dataset):
    def __init__(self, games_dir, max_workers=15, use_processes=True):
        self.inputs = []
        self.outputs = []
        
        # Get all game files
        files = glob.glob(games_dir + "/games*")

        # Choose executor based on preference
        executor_class = ProcessPoolExecutor if use_processes else ThreadPoolExecutor

        # Use executor to load files in parallel
        with executor_class(max_workers=max_workers) as executor:
            # Submit each file for processing
            futures = [executor.submit(self._process_file, filename) for filename in files]

            # Collect results as they complete
            for future in futures:
                file_inputs, file_outputs = future.result()
                self.inputs.extend(file_inputs)
                self.outputs.extend(file_outputs)                

        mem = self.get_memory_usage()
        print(f"Inputs:  {mem['inputs_mb']:.2f} MB")
        print(f"Outputs: {mem['outputs_mb']:.2f} MB")
        print(f"Total:   {mem['total_mb']:.2f} MB")
        

    def _process_file(self, filename):
        """Process a single file and return (inputs, outputs)"""
        inputs = []
        outputs = []

        print(filename)
        with open(filename, 'rb') as f:
            data = f.read()
            print(len(data))
            counter = 1
            while data:
                if counter % 1000 == 0:
                    print(counter)
                counter += 1

                nb_moves = int(data[0])
                d = data[:nb_moves * 2 + 4]
                nb_random_moves = int(d[1])

                if nb_random_moves == nb_moves:
                    data = data[2 + 2*nb_moves + 2:]
                    continue

                black_score = int(d[-2])
                white_score = int(d[-1])
                res = 0
                if black_score == white_score:
                    res = 1
                if white_score > black_score:
                    res = 2
                res = int(res)  # Keep as Python int, not tensor

                yolah = Yolah()
                for sq1, sq2 in zip(d[2:2+2*nb_random_moves:2], data[3:2+2*nb_random_moves:2]):
                    m = Move(Square(int(sq1)), Square(int(sq2)))
                    yolah.play(m)

                inputs.append(self.encode_yolah(yolah))  # Returns numpy array
                outputs.append(res)

                for sq1, sq2 in zip(d[2+2*nb_random_moves:2+2*nb_moves:2], d[3+2*nb_random_moves:2+2*nb_moves:2]):
                    m = Move(Square(int(sq1)), Square(int(sq2)))
                    yolah.play(m)
                    inputs.append(self.encode_yolah(yolah))  # Returns numpy array
                    outputs.append(res)

                data = data[2 + 2*nb_moves + 2:]

        #print(sys.getsizeof(inputs) / (1024 * 1024))
        return inputs, outputs

    @staticmethod
    def encode_yolah(yolah):
        black_list = bitboard64_to_list(yolah.black)
        white_list = bitboard64_to_list(yolah.white)
        one_hot = np.array(list(itertools.chain.from_iterable([
                        black_list,
                        white_list,
                        bitboard64_to_list(yolah.empty),
                        [Yolah.WHITE_PLAYER if yolah.nb_plies() & 1 else Yolah.BLACK_PLAYER]])), dtype=np.uint8)
        return one_hot
    
    def __len__(self):
        return len(self.outputs)

    def __getitem__(self, idx):
        return (self.inputs[idx], self.outputs[idx])

    def get_memory_usage(self):
        """Returns memory usage in bytes and MB"""
        # Calculate inputs memory (numpy arrays)
        inputs_bytes = self.inputs[0].nbytes * len(self.inputs)
        # for arr in self.inputs:
        #     inputs_bytes += arr.nbytes  # NumPy array memory size

        # Calculate outputs memory (Python ints)
        outputs_bytes = sys.getsizeof(self.outputs[0]) * len(self.outputs)
        # for val in self.outputs:
        #     outputs_bytes += sys.getsizeof(val)  # Python int object size

        total_bytes = inputs_bytes + outputs_bytes

        return {
            'inputs_mb': inputs_bytes / (1024**2),
            'outputs_mb': outputs_bytes / (1024**2),
            'total_mb': total_bytes / (1024**2),
            'inputs_bytes': inputs_bytes,
            'outputs_bytes': outputs_bytes,
            'total_bytes': total_bytes
        }

# black positions + white positions + empty positions + turn
INPUT_SIZE = 64 + 64 + 64 + 1

class GameDatasetMemmap(Dataset):
    """
    Memory-mapped dataset that efficiently shares data across multiple worker processes.
    All workers access the same on-disk data without copying it to each process.

    This is ideal for distributed training where you want to avoid RAM duplication
    across worker processes.
    """
    def __init__(self, data_prefix):
        """
        Args:
            data_prefix: Path prefix for memmap files (e.g., "data/dataset")
                        Will look for "{data_prefix}_inputs.npy" and "{data_prefix}_outputs.npy"
        """
        inputs_path = f'{data_prefix}_inputs.npy'
        outputs_path = f'{data_prefix}_outputs.npy'

        if not os.path.exists(inputs_path) or not os.path.exists(outputs_path):
            raise FileNotFoundError(
                f"Memory-mapped files not found. Please run preprocessing first:\n"
                f"  from nnue_multigpu3 import preprocess_dataset_to_memmap\n"
                f"  preprocess_dataset_to_memmap('data', 'data/dataset')"
            )

        # Calculate number of samples from file size
        file_size_bytes = os.path.getsize(inputs_path)
        # Each sample is INPUT_SIZE bytes (uint8 = 1 byte per element)
        num_samples = file_size_bytes // INPUT_SIZE

        # Open as read-only memory-mapped arrays
        # This is shared across all processes without duplication
        self.inputs = np.memmap(inputs_path, dtype=np.uint8, mode='r', shape=(num_samples, INPUT_SIZE))
        self.outputs = np.memmap(outputs_path, dtype=np.uint8, mode='r')

        print(f"Loaded memory-mapped dataset from {data_prefix}")
        print(f"  Total samples: {len(self)}")
        print(f"  Inputs shape:  {self.inputs.shape}")
        print(f"  Outputs shape: {self.outputs.shape}")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        """
        Get a single sample. Memory-map handles efficient access for concurrent reads.
        """
        # Direct access - no copy needed since we only read (no data race)
        input_data = self.inputs[idx]
        output_data = self.outputs[idx]

        return torch.from_numpy(input_data.astype(np.float32)), torch.tensor(output_data, dtype=torch.long)

# def custom_collate(batch, vocab_size=INPUT_SIZE):
#     """
#     Convert batch of sparse indices to dense one-hot tensors.
#     Each sample is a tuple of (sparse_indices, target).
#     """
#     inputs, targets = zip(*batch)
#     batch_size = len(inputs)
#     device = inputs[0].device

#     # Create dense tensor for the batch
#     dense = torch.zeros(batch_size, vocab_size, dtype=torch.float32, device=device, pin_memory=True)

#     # Fill in the active features for each sample in the batch
#     for i, sparse_indices in enumerate(inputs):
#         # Convert uint8 to long for indexing
#         indices = sparse_indices.long()
#         dense[i, indices] = 1.0

#     return dense, torch.stack(targets)

class Net(nn.Module):
    def __init__(self, input_size=INPUT_SIZE, l1_size=1024, l2_size=64, l3_size=32):
        super().__init__()
        self.fc1 = nn.Linear(input_size, l1_size)
        self.fc2 = nn.Linear(l1_size, l2_size)
        self.fc3 = nn.Linear(l2_size, l3_size)
        self.fc4 = nn.Linear(l3_size, 3)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.clamp(x, min=0.0, max=1.0)
        x = self.fc2(x)
        x = torch.clamp(x, min=0.0, max=1.0)
        x = self.fc3(x)
        x = torch.clamp(x, min=0.0, max=1.0)
        return self.fc4(x)#softmax(self.fc4(x), dim=1)#

NB_EPOCHS=200
#MODEL_PATH="./"
MODEL_PATH="/mnt/"
MODEL_NAME="nnue_1024x64x32x3_2"
LAST_MODEL=f"{MODEL_PATH}{MODEL_NAME}.pt"

def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "65432"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

def dataloader_ddp(trainset, valset, batch_size):
    sampler_train = DistributedSampler(trainset)
    sampler_val = DistributedSampler(valset, shuffle=False)
    # train_loader = DataLoader(
    #     trainset, batch_size=batch_size, shuffle=False, sampler=sampler_train,
    #     num_workers=2, pin_memory=True, prefetch_factor=2, persistent_workers=False#, collate_fn=custom_collate
    # )
    # val_loader = DataLoader(
    #     valset, batch_size=batch_size, shuffle=False, sampler=sampler_val,
    #     num_workers=2, pin_memory=True, prefetch_factor=2, persistent_workers=False#, collate_fn=custom_collate
    # )
    train_loader = DataLoader(
        trainset, batch_size=batch_size, shuffle=False, sampler=sampler_train,
        num_workers=0, pin_memory=True#, collate_fn=custom_collate
    )
    val_loader = DataLoader(
        valset, batch_size=batch_size, shuffle=False, sampler=sampler_val,
        num_workers=0, pin_memory=True#, collate_fn=custom_collate
    )
    return train_loader, sampler_train, val_loader, sampler_val

class TrainerDDP:
    def __init__(self, gpu_id, model, train_loader, sampler_train, val_loader, sampler_val, save_every=10):
        self.gpu_id = gpu_id
        self.model = model.to(self.gpu_id)
        self.train_loader = train_loader
        self.sampler_train = sampler_train
        self.val_loader = val_loader
        self.sampler_val = sampler_val
        self.save_every = save_every
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=0)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        torch.cuda.set_device(gpu_id)
        torch.cuda.empty_cache()
        self.model = DDP(self.model, device_ids=[self.gpu_id])
        self.model = torch.compile(self.model)
        # Create CUDA stream for asynchronous data transfer
        self.stream = torch.cuda.Stream(device=gpu_id)
        
    def _save_checkpoint(self, epoch):
        torch.save(self.model.module.state_dict(), f"{MODEL_PATH}{MODEL_NAME}.{epoch}.pt")

    def _run_epoch(self, epoch):
        n = 0
        running_loss = 0
        accuracy = 0
        for (X, y) in self.train_loader:
            n += len(X)
            # X = X.to(self.gpu_id)
            # y = y.to(self.gpu_id)
            # Use CUDA stream for asynchronous data transfer
            with torch.cuda.stream(self.stream):
                X = X.to(self.gpu_id, non_blocking=True)
                y = y.to(self.gpu_id, non_blocking=True)
            # Ensure transfer is complete before computation
            torch.cuda.current_stream(self.gpu_id).wait_stream(self.stream)

            self.optimizer.zero_grad()
            logits = self.model(X)
            #y = y.to(self.gpu_id, dtype=torch.long)
            loss = self.loss_fn(logits, y)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            accuracy += sum(torch.argmax(logits, dim=1) == y).item()
        if self.gpu_id == 0:
            print('epoch {} train loss: {} train accuracy: {}'.format(epoch + 1, running_loss / n, accuracy / n), flush=True)

    def _validate(self, epoch):
        self.model.eval()
        n = 0
        val_loss = 0
        val_accuracy = 0
        with torch.no_grad():
            for (X, y) in self.val_loader:
                n += len(X)
                # X = X.to(self.gpu_id)
                # y = y.to(self.gpu_id)
                # Use CUDA stream for asynchronous data transfer
                with torch.cuda.stream(self.stream):
                    X = X.to(self.gpu_id, non_blocking=True)
                    y = y.to(self.gpu_id, non_blocking=True)
                # # Ensure transfer is complete before computation
                torch.cuda.current_stream(self.gpu_id).wait_stream(self.stream)

                logits = self.model(X)
                #y = y.to(self.gpu_id, dtype=torch.long)
                loss = self.loss_fn(logits, y)
                val_loss += loss.item()
                val_accuracy += sum(torch.argmax(logits, dim=1) == y).item()
        if self.gpu_id == 0:
            print('epoch {} val loss: {} val accuracy: {}'.format(epoch + 1, val_loss / n, val_accuracy / n), flush=True)
        self.model.train()

    def train(self, nb_epochs):
        self.model.train()
        for epoch in range(nb_epochs):
            self.sampler_train.set_epoch(epoch)
            self._run_epoch(epoch)
            self._validate(epoch)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch)
        self._save_checkpoint(nb_epochs - 1)

def main(rank, world_size, batch_size, dataset):
    ddp_setup(rank, world_size)
    print(rank)
    if rank == 0:
        print(len(dataset), flush=True)

    # Split dataset into train and validation (90/10 split)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    trainset, valset = random_split(dataset, [train_size, val_size])

    if rank == 0:
        print(f'Train size: {train_size}, Val size: {val_size}', flush=True)

    train_loader, sampler_train, val_loader, sampler_val = dataloader_ddp(trainset, valset, batch_size)
    net = Net()
    if os.path.isfile(LAST_MODEL):
        net.load_state_dict(torch.load(LAST_MODEL))
    if rank == 0:
        print(net, flush=True)
    trainer = TrainerDDP(rank, net, train_loader, sampler_train, val_loader, sampler_val)
    trainer.train(NB_EPOCHS)
    destroy_process_group()

if __name__ == "__main__":
    print(torch.cuda.is_available())
    world_size = torch.cuda.device_count()
    print(world_size, flush=True)

    # Dataset configuration
    DATA_PREFIX = "/mnt/"  # Path prefix for memory-mapped files

    # Check if preprocessing is needed
    if not (os.path.exists(f'{DATA_PREFIX}_inputs.npy') and os.path.exists(f'{DATA_PREFIX}_outputs.npy')):
        print("Memory-mapped dataset not found. Running preprocessing...", flush=True)
        print("This will create:", flush=True)
        print(f"  - {DATA_PREFIX}_inputs.npy", flush=True)
        print(f"  - {DATA_PREFIX}_outputs.npy", flush=True)
        preprocess_dataset_to_memmap("/nnue/data/", DATA_PREFIX)
        print("Preprocessing complete!", flush=True)
    else:
        print("Using existing memory-mapped dataset", flush=True)

    # Load dataset from memory-mapped files
    print("Loading memory-mapped dataset...", flush=True)
    dataset = GameDatasetMemmap(DATA_PREFIX)
    print("Dataset loaded successfully!", flush=True)

    # Spawn workers with the shared dataset
    # Workers will access the same memory-mapped files without duplication
    mp.spawn(main, args=(world_size, 16384 * 2, dataset), nprocs=world_size)
