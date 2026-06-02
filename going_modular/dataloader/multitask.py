import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import numpy as np
import os
import pandas as pd
import cv2

class PhotometricDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, type_mode='albedo', id_to_label=None):
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
        if id_to_label is not None:
            self.unique_ids  = sorted(id_to_label.keys())
            self.id_to_label = id_to_label
        else:
            self.unique_ids  = sorted(self.df['id'].unique())
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
    def __init__(self, csv_file, root_dir, transform=None, id_to_label=None):
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

        if id_to_label is not None:
            self.unique_ids  = sorted(id_to_label.keys())
            self.id_to_label = id_to_label
        else:
            self.unique_ids  = sorted(self.df['id'].unique())
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


def create_multitask_datafetcher(config, train_transform, test_transform, file_train='train_set.csv', file_test='test_set.csv'):
    dataset_dir = config['dataset_dir']

    train_csv = os.path.join(dataset_dir, file_train)
    test_csv  = os.path.join(dataset_dir, file_test)

    # Kaggle dataset tách ảnh theo subfolder: Albedo/, Normal_Map/, Depth_Map/
    # Nếu subfolder tồn tại thì dùng, ngược lại fallback về dataset_dir (cấu trúc cũ)
    type_mode = config.get('type', 'albedo')
    _subdir_map = {'albedo': 'Albedo', 'normalmap': 'Normal_Map', 'depthmap': 'Depth_Map'}
    _subdir = os.path.join(dataset_dir, _subdir_map.get(type_mode, ''))
    image_root = _subdir if os.path.isdir(_subdir) else dataset_dir

    train_ds = PhotometricDataset(train_csv, image_root, train_transform, type_mode)
    test_ds  = PhotometricDataset(test_csv,  image_root, test_transform,  type_mode)

    use_sampler = config.get('use_sampler', False)
    if use_sampler:
        print(f">>> SingleLoader: MODE = PK SAMPLER")
        train_dl = DataLoader(train_ds, batch_sampler=UniqueIdBatchSampler(train_ds.get_labels(), config['batch_size']), num_workers=2)
    else:
        print(f">>> SingleLoader: MODE = RANDOM SHUFFLE")
        train_dl = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True, num_workers=2)

    test_dl = DataLoader(test_ds, batch_size=config['batch_size'], shuffle=False, num_workers=2)
    return train_dl, test_dl, train_ds.weightclass


def create_concatv2_multitask_datafetcher(config, train_transform, test_transform, file_train='train_set.csv', file_test='test_set.csv'):
    dataset_dir = config['dataset_dir']
    train_csv = os.path.join(dataset_dir, file_train)
    if not os.path.exists(train_csv): train_csv = os.path.join(dataset_dir, 'dataset', 'train_set.csv')
    test_csv = os.path.join(dataset_dir, file_test)
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


def create_eval_loaders(
    config,
    transform,
    gallery_csv_name: str = 'gallery_split.csv',
    probe_csv_name:   str = 'probe_split.csv',
):
    """
    Tạo gallery DataLoader (reference) và probe DataLoader (query) với shared identity mapping.

    Hai dataset dùng cùng id_to_label nên label comparison gallery vs probe luôn nhất quán,
    kể cả khi hai split có tập identity không hoàn toàn giống nhau.
    """
    dataset_dir = config['dataset_dir']
    type_mode   = config.get('type', 'albedo')
    batch_size  = config.get('batch_size', 32)

    _subdir_map = {'albedo': 'Albedo', 'normalmap': 'Normal_Map', 'depthmap': 'Depth_Map'}
    _subdir    = os.path.join(dataset_dir, _subdir_map.get(type_mode, ''))
    image_root = _subdir if os.path.isdir(_subdir) else dataset_dir

    gallery_csv = os.path.join(dataset_dir, gallery_csv_name)
    probe_csv   = os.path.join(dataset_dir, probe_csv_name)

    if not os.path.exists(gallery_csv):
        raise FileNotFoundError(f'Gallery CSV không tìm thấy: {gallery_csv}')
    if not os.path.exists(probe_csv):
        raise FileNotFoundError(f'Probe CSV không tìm thấy: {probe_csv}')

    gdf = pd.read_csv(gallery_csv)
    pdf = pd.read_csv(probe_csv)
    all_ids       = sorted(set(gdf['id'].unique()) | set(pdf['id'].unique()))
    shared_id_map = {id_val: i for i, id_val in enumerate(all_ids)}

    if type_mode == 'concat_v2':
        gallery_ds = ConcatCustomExrDatasetV2(gallery_csv, dataset_dir, transform, id_to_label=shared_id_map)
        probe_ds   = ConcatCustomExrDatasetV2(probe_csv,   dataset_dir, transform, id_to_label=shared_id_map)
    else:
        gallery_ds = PhotometricDataset(gallery_csv, image_root, transform, type_mode, id_to_label=shared_id_map)
        probe_ds   = PhotometricDataset(probe_csv,   image_root, transform, type_mode, id_to_label=shared_id_map)

    gallery_dl = DataLoader(gallery_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    probe_dl   = DataLoader(probe_ds,   batch_size=batch_size, shuffle=False, num_workers=2)

    print(f'Gallery: {len(gallery_ds)} ảnh | Probe: {len(probe_ds)} ảnh')
    print(f'Shared identity space: {len(shared_id_map)} identities')
    return gallery_dl, probe_dl