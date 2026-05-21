import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import numpy as np
import os
import pandas as pd
import cv2

class PhotometricDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, type_mode='albedo'):
        if not os.path.exists(csv_file):
            alt_csv = os.path.join(os.path.dirname(csv_file), 'dataset', os.path.basename(csv_file))
            if os.path.exists(alt_csv): csv_file = alt_csv
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.type_mode = type_mode
        self.file_map = {
            'albedo': 'albedo_map_new_crop.exr.npy',
            'normalmap': 'normal_map_new_crop.exr.npy',
            'depthmap': 'depth_map_new_crop.exr.npy',
        
        }
        self.unique_ids = sorted(self.df['id'].unique())
        self.id_to_label = {id_val: i for i, id_val in enumerate(self.unique_ids)}
        self.labels_list = [self.id_to_label[row['id']] for _, row in self.df.iterrows()]
        self.weightclass = {}

    def __len__(self): return len(self.df)
    def get_labels(self): return self.labels_list

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        id_val = row['id']
        session_name = row['session']
        file_suffix = self.file_map.get(self.type_mode, 'albedo_map_new_crop.exr.npy')
        file_path = os.path.join(self.root_dir, str(id_val), str(session_name), file_suffix)
        try:
            image_data = np.load(file_path)
            if image_data.ndim == 3 and image_data.shape[0] == 3:
                image_data = image_data.transpose(1, 2, 0)
            image_data = image_data.astype(np.float32)
        except:
            image_data = np.zeros((112, 112, 3), dtype=np.float32)

        label_id = self.id_to_label[id_val]
        def get_lbl(key): return int(row.get(key, 0))
        labels = torch.tensor([label_id, get_lbl('Gender'), get_lbl('Spectacles'), get_lbl('Facial_Hair'), get_lbl('Pose'), get_lbl('Emotion')], dtype=torch.long)

        if self.transform:
            res = self.transform(image=image_data)
            image_data = res['image']

        if isinstance(image_data, np.ndarray):
            X = torch.from_numpy(image_data).permute(2, 0, 1)
        else: X = image_data
        return X, labels


class ConcatCustomExrDatasetV2(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        if not os.path.exists(csv_file):
            alt_csv = os.path.join(os.path.dirname(csv_file), os.path.basename(csv_file))
            if os.path.exists(alt_csv): csv_file = alt_csv
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

        # Map filenames
        self.file_map = {
            'albedo': 'albedo_map_new_crop.exr.npy',
            'normal': 'normal_map_new_crop.exr.npy'
        }

        self.unique_ids = sorted(self.df['id'].unique())
        self.id_to_label = {id_val: i for i, id_val in enumerate(self.unique_ids)}
        self.labels_list = [self.id_to_label[row['id']] for _, row in self.df.iterrows()]

    def __len__(self): return len(self.df)
    def get_labels(self): return self.labels_list

    def __load_npy(self, path):
        try:
            if not os.path.exists(path): return None
            img = np.load(path)
            if img.ndim == 3 and img.shape[0] == 3: img = img.transpose(1, 2, 0)
            return img.astype(np.float32)
        except: return None

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        id_val = row['id']
        session = str(row['session'])

        p1 = os.path.join(self.root_dir, str(id_val), session, self.file_map['albedo'])
        p2 = os.path.join(self.root_dir, str(id_val), session, self.file_map['normal'])

        i1 = self.__load_npy(p1)
        i2 = self.__load_npy(p2)

        if i1 is None: i1 = np.zeros((112, 112, 3), dtype=np.float32)
        if i2 is None: i2 = np.zeros((112, 112, 3), dtype=np.float32) #

        
        if i2.shape[:2] != i1.shape[:2]: i2 = cv2.resize(i2, (i1.shape[1], i1.shape[0]))

        label_id = self.id_to_label[id_val]
        def get_lbl(key): return int(row.get(key, 0))
        labels = torch.tensor([label_id, get_lbl('Gender'), get_lbl('Spectacles'), get_lbl('Facial_Hair'), get_lbl('Pose'), get_lbl('Emotion')], dtype=torch.long)

        if self.transform:
            res = self.transform(image=i1, image2=i2)
            i1, i2 = res['image'], res['image2']

        to_ts = lambda x: torch.from_numpy(x).permute(2, 0, 1) if isinstance(x, np.ndarray) else x

   
        X = torch.stack((to_ts(i1), to_ts(i2)), dim=0)
        return X, labels


class UniqueIdBatchSampler(Sampler):
    def __init__(self, labels, batch_size):
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.unique_labels = list(set(labels))
        self.label_indices = {l: [] for l in self.unique_labels}
        for idx, l in enumerate(self.labels): self.label_indices[l].append(idx)
    def __iter__(self):
        n_batches = len(self.labels) // self.batch_size
        for _ in range(n_batches):
            batch_labels = np.random.choice(self.unique_labels, size=self.batch_size, replace=False)
            batch_indices = [np.random.choice(self.label_indices[l]) for l in batch_labels]
            yield batch_indices
    def __len__(self): return len(self.labels) // self.batch_size


def create_multitask_datafetcher(config, train_transform, test_transform):
    dataset_dir = config['dataset_dir']

    train_csv = os.path.join(dataset_dir, 'train_set.csv')
    if not os.path.exists(train_csv): train_csv = os.path.join(dataset_dir, 'train_set.csv')
    test_csv = os.path.join(dataset_dir, 'test_set.csv')
    if not os.path.exists(test_csv): test_csv = os.path.join(dataset_dir, 'test_set.csv')

    train_ds = PhotometricDataset(train_csv, dataset_dir, train_transform, config.get('type', 'albedo'))
    test_ds = PhotometricDataset(test_csv, dataset_dir, test_transform, config.get('type', 'albedo'))

    use_sampler = config.get('use_sampler', False)
    if use_sampler:
        print(f">>> SingleLoader: MODE = PK SAMPLER")
        train_dl = DataLoader(train_ds, batch_sampler=UniqueIdBatchSampler(train_ds.get_labels(), config['batch_size']), num_workers=2)
    else:
        print(f">>> SingleLoader: MODE = RANDOM SHUFFLE")
        train_dl = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True, num_workers=2)

    test_dl = DataLoader(test_ds, batch_size=config['batch_size'], shuffle=False, num_workers=2)
    return train_dl, test_dl, train_ds.weightclass


def create_concatv2_multitask_datafetcher(config, train_transform, test_transform):
    dataset_dir = config['dataset_dir']
    train_csv = os.path.join(dataset_dir, 'train_set.csv')
    if not os.path.exists(train_csv): train_csv = os.path.join(dataset_dir, 'dataset', 'train_set.csv')
    test_csv = os.path.join(dataset_dir, 'test_set.csv')
    if not os.path.exists(test_csv): test_csv = os.path.join(dataset_dir, 'dataset', 'test_set.csv')

  
    train_ds = ConcatCustomExrDatasetV2(train_csv, dataset_dir, train_transform)
    test_ds = ConcatCustomExrDatasetV2(test_csv, dataset_dir, test_transform)

    use_sampler = config.get('use_sampler', False)
    if use_sampler:
        print(f">>> ConcatV2Loader: MODE = PK SAMPLER")
        train_dl = DataLoader(train_ds, batch_sampler=UniqueIdBatchSampler(train_ds.get_labels(), config['batch_size']), num_workers=2)
    else:
        print(f">>> ConcatV2Loader: MODE = RANDOM SHUFFLE")
        train_dl = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True, num_workers=2)

    test_dl = DataLoader(test_ds, batch_size=config['batch_size'], shuffle=False, num_workers=2)
    return train_dl, test_dl