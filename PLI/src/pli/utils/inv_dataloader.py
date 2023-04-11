import glob
import random

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split

from ..utilsdata.utils import NumpyDataset, worker_init_fn

PAIR_SIZE_MAX = 100


class NumpyAEDataset(NumpyDataset):
    """This class allows you to convert numpy.array to torch.Dataset
    Args:
        x (np.array):
        y (np.array):
        transform (torch.transform):
    Attriutes
        x (np.array):
        y (np.array):
        transform (torch.transform):
    """

    def __init__(self, x, y=None, transform=None, return_idx=False):
        self.x = x
        self.y = y
        self.transform = transform
        self.return_idx = return_idx

    def __getitem__(self, index):
        x = self.x[index]
        if self.y is not None:
            y = self.y[index]

        if self.transform is not None:
            x = self.transform(x)
            y = self.transform(y)

        if not self.return_idx:
            if self.y is not None:
                return x, y
            else:
                return x
        else:
            if self.y is not None:
                return index, x, y
            else:
                return index, x

    def __len__(self):
        """get the number of rows of self.x"""
        return len(self.x)


def prepare_inv_lag_dataloaders(
    data_folder="../input/large-agegap",
    client_num=2,
    batch_size=1,
    channel=1,
    seed=42,
    num_workers=2,
    height=64,
    width=64,
    num_classes=1010,
    crop=True,
    target_celeblities_num=100,
):
    """Prepare dataloaders for LAG dataset.

    Args:
        data_folder (str, optional): a path to the data folder. Defaults to "/content".
        client_num (int, optional): the number of clients. Defaults to 2.
        channel (int, optional): the number of channes. Defaults to 1.
        batch_size (int, optional): batch size for training. Defaults to 1.
        seed (int, optional): seed of randomness. Defaults to 42.
        num_workers (int, optional): the number of workers. Defaults to 2.
        height (int, optional): height of images. Defaults to 64.
        width (int, optional): width of images. Defaults to 64.
        num_classes (int, optional): the number of classes. Defaults to 1000.
        crop (bool, optional): crop the image to (height, width) if true. Defaults to True.
        target_celeblities_num (int, optional): the number of target labels. Defaults to 100.

    Returns:
        a tuple of public dataloader, a list of local dataloaders, test dataloader,
        and the list of target labels.
    """
    paths = glob.glob(f"{data_folder}/*")
    paths.sort()

    name_list = []
    path_list = []
    ay_list = []

    for p in paths:
        name = p.split("/")[-1]
        if name == "README.txt":
            continue
        a_paths = glob.glob(f"{p}/*.*")
        y_paths = glob.glob(f"{p}/y/*.*")

        name_list += [name for _ in range(len(a_paths))]
        name_list += [name for _ in range(len(y_paths))]
        path_list += a_paths
        path_list += y_paths
        ay_list += [1 for _ in range(len(a_paths))]
        ay_list += [0 for _ in range(len(y_paths))]
    df = pd.DataFrame(columns=["name", "path", "ay"])
    df["name"] = name_list
    df["path"] = path_list
    df["ay"] = ay_list

    top_identities = (
        df.groupby("name")
        .count()
        .sort_values("ay", ascending=False)
        .index[:num_classes]
    )
    df["top"] = df["name"].apply(lambda x: x in top_identities)
    df = df[df["top"]]

    unique_name_list = []
    unique_name_min_img_num = []

    for name in df["name"].unique():
        unique_name_list.append(name)
        unique_name_min_img_num.append(
            df[df["name"] == name].groupby("ay").count().min()["name"]
        )
    unique_name_list = np.array(unique_name_list)
    name2id = {name: i for i, name in enumerate(unique_name_list)}
    unique_name_min_img_num = np.array(unique_name_min_img_num)

    local_identities = random.sample(list(unique_name_list), target_celeblities_num)
    local_identities = np.array_split(local_identities, client_num)
    local_identities = [id_list.tolist() for id_list in local_identities]

    alloc = [-1 for _ in range(df.shape[0])]
    for j, (ay, name) in enumerate(zip(df["ay"].tolist(), df["name"].tolist())):
        if ay == 1:
            for i in range(client_num):
                if name in local_identities[i]:
                    alloc[j] = i + 1
                    break
        if alloc[j] == -1:
            alloc[j] = 0
    df["alloc"] = alloc

    X_public = np.stack([cv2.imread(p) for p in df[df["alloc"] == 0]["path"].tolist()])
    is_sensitive_public = df[df["alloc"] == 0]["ay"].values
    y_public = np.array([name2id[n] for n in df[df["alloc"] == 0]["name"].tolist()])

    transforms_list = [transforms.ToTensor()]
    if channel == 1:
        transforms_list.append(transforms.Grayscale())
    if crop:
        transforms_list.append(transforms.CenterCrop((max(height, width))))
    else:
        transforms_list.append(transforms.Resize((height, width)))
    if channel == 1:
        transforms_list.append(transforms.Normalize((0.5,), (0.5,)))
    else:
        transforms_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    transform = transforms.Compose(transforms_list)
    return_idx = True

    sensitive_idx = np.where(is_sensitive_public == 1)[0]
    nonsensitive_idx = np.where(is_sensitive_public == 0)[0]
    X_public_input_inv = []
    X_public_output_inv = []

    skipped = 0
    for y in list(np.unique(y_public)):
        y_idx = np.where(y_public == y)[0]
        y_sensitive_idx = list(set(list(y_idx)) & set(list(sensitive_idx)))
        y_nonsensitive_idx = list(set(list(y_idx)) & set(list(nonsensitive_idx)))

        pairs = sum(
            [[(ys, yn) for yn in y_nonsensitive_idx] for ys in y_sensitive_idx], []
        )
        pairs = random.sample(pairs, min(PAIR_SIZE_MAX, len(pairs)))

        if len(pairs) == 0:
            skipped += 1

        for pair in pairs:
            X_public_input_inv.append(X_public[pair[1]])
            X_public_output_inv.append(X_public[pair[0]])

    print("#skipped", skipped)

    X_public_input_inv = np.stack(X_public_input_inv)
    X_public_output_inv = np.stack(X_public_output_inv)

    public_inv_dataset = NumpyAEDataset(
        x=X_public_input_inv,
        y=X_public_output_inv,
        transform=transform,
        return_idx=return_idx,
    )

    g = torch.Generator()
    g.manual_seed(seed)

    print("prepare public dataloader")
    try:
        public_inv_dataloader = torch.utils.data.DataLoader(
            public_inv_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            worker_init_fn=worker_init_fn,
            generator=g,
        )
    except:
        public_inv_dataloader = torch.utils.data.DataLoader(
            public_inv_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            worker_init_fn=worker_init_fn,
        )

    return public_inv_dataloader


def prepare_inv_lfw_dataloaders(
    data_folder="/content",
    client_num=2,
    channel=1,
    batch_size=1,
    seed=42,
    num_workers=2,
    height=64,
    width=64,
    num_classes=1000,
    crop=True,
    target_celeblities_num=100,
):
    """Prepare dataloaders for LFW dataset.

    Args:
        data_folder (str, optional): a path to the data folder. Defaults to "/content".
        client_num (int, optional): the number of clients. Defaults to 2.
        channel (int, optional): the number of channes. Defaults to 1.
        batch_size (int, optional): batch size for training. Defaults to 1.
        seed (int, optional): seed of randomness. Defaults to 42.
        num_workers (int, optional): the number of workers. Defaults to 2.
        height (int, optional): height of images. Defaults to 64.
        width (int, optional): width of images. Defaults to 64.
        num_classes (int, optional): the number of classes. Defaults to 1000.
        crop (bool, optional): crop the image to (height, width) if true. Defaults to True.
        target_celeblities_num (int, optional): the number of target labels. Defaults to 100.

    Returns:
        a tuple of public dataloader, a list of local dataloaders, test dataloader,
        and the list of target labels.
    """
    nomask_path_list = glob.glob(f"{data_folder}/lfw-align-128/*/*")
    nomask_path_list.sort()
    mask_path_list = glob.glob(f"{data_folder}/lfw-align-128-masked/*/*")
    mask_path_list.sort()

    path_list = []
    nomask_path_list = []
    name_list = []
    # ismask_list = []
    for mask_path in mask_path_list:
        name = mask_path.split("/")[-2]
        file_name = mask_path.split("/")[-1]
        nomask_path = f"{data_folder}/lfw-align-128/{name}/{file_name}"
        name_list.append(name)
        path_list.append(mask_path)
        nomask_path_list.append(nomask_path)
        # if random.random() > 0.5:
        #    path_list.append(mask_path)
        #    ismask_list.append(1)
        # else:
        #    path_list.append(nomask_path)
        #    ismask_list.append(0)

    df = pd.DataFrame(columns=["name", "path", "nomask_path", "ismask"])
    df["name"] = name_list
    df["path"] = path_list
    df["nomask_path"] = nomask_path_list
    # df["ismask"] = ismask_list

    top_identities = (
        df.groupby("name")
        .count()
        .sort_values("path", ascending=False)
        .index[:num_classes]
        .to_list()
    )
    df["top"] = df["name"].apply(lambda x: x in top_identities)
    df = df[df["top"]]

    # name_with_both_types_of_images = []
    # for name in df["name"].unique():
    #    if df[df["name"] == name].groupby("ismask").count().shape[0] > 1:
    #        name_with_both_types_of_images.append(name)

    name2id = {name: i for i, name in enumerate(top_identities)}

    local_identities_names = np.array_split(
        random.sample(top_identities, target_celeblities_num),
        client_num,
    )
    local_identities = [
        [name2id[name] for name in name_list] for name_list in local_identities_names
    ]

    name_id2client_id = {}
    for client_id, name_id_list in enumerate(local_identities):
        for idx in name_id_list:
            name_id2client_id[idx] = client_id

    X_public_list = []
    y_public_list = []
    is_sensitive_public_list = []
    X_private_lists = [[] for _ in range(client_num)]
    y_private_lists = [[] for _ in range(client_num)]

    for u_name in top_identities:
        idxs = np.array_split(list(range(df[df["name"] == u_name].shape[0])), 2)
        path_list = df[df["name"] == u_name]["path"].values
        nomask_path_list = df[df["name"] == u_name]["nomask_path"].values

        for idx in idxs:
            for path in path_list[idx]:
                X_public_list.append(cv2.imread(path))
                y_public_list.append(name2id[u_name])
                is_sensitive_public_list.append(0)
            for path in nomask_path_list[idx]:
                if name2id[u_name] in name_id2client_id:
                    X_private_lists[name_id2client_id[name2id[u_name]]].append(
                        cv2.imread(path)
                    )
                    y_private_lists[name_id2client_id[name2id[u_name]]].append(
                        name2id[u_name]
                    )
                else:
                    X_public_list.append(cv2.imread(path))
                    y_public_list.append(name2id[u_name])
                    is_sensitive_public_list.append(1)

    X_public = np.stack(X_public_list)
    y_public = np.array(y_public_list)
    is_sensitive_public = np.array(is_sensitive_public_list)
    # X_private_list = [np.stack(x) for x in X_private_lists]
    y_private_list = [np.array(y) for y in y_private_lists]

    print("#nonsensitive labels: ", len(np.unique(y_public)))
    print(
        "#sensitive labels: ",
        len(np.unique(sum([t.tolist() for t in y_private_list], []))),
    )

    transforms_list = [transforms.ToTensor()]
    if channel == 1:
        transforms_list.append(transforms.Grayscale())
    if crop:
        transforms_list.append(transforms.CenterCrop((max(height, width))))
    else:
        transforms_list.append(transforms.Resize((height, width)))
    if channel == 1:
        transforms_list.append(transforms.Normalize((0.5,), (0.5,)))
    else:
        transforms_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    transform = transforms.Compose(transforms_list)
    return_idx = True

    sensitive_idx = np.where(is_sensitive_public == 1)[0]
    nonsensitive_idx = np.where(is_sensitive_public == 0)[0]
    X_public_input_inv = []
    X_public_output_inv = []

    skipped = 0
    sensitive_labels = np.unique(sum([t.tolist() for t in y_private_list], [])).tolist()
    for y in list(np.unique(y_public)):
        if y not in sensitive_labels:
            y_idx = np.where(y_public == y)[0]
            y_sensitive_idx = list(set(list(y_idx)) & set(list(sensitive_idx)))
            y_nonsensitive_idx = list(set(list(y_idx)) & set(list(nonsensitive_idx)))

            pairs = sum(
                [[(ys, yn) for yn in y_nonsensitive_idx] for ys in y_sensitive_idx], []
            )
            pairs = random.sample(pairs, min(PAIR_SIZE_MAX, len(pairs)))

            if len(pairs) == 0:
                skipped += 1

            for pair in pairs:
                X_public_input_inv.append(X_public[pair[1]])
                X_public_output_inv.append(X_public[pair[0]])

    print("#skipped", skipped)

    X_public_input_inv = np.stack(X_public_input_inv)
    X_public_output_inv = np.stack(X_public_output_inv)

    public_inv_dataset = NumpyAEDataset(
        x=X_public_input_inv,
        y=X_public_output_inv,
        transform=transform,
        return_idx=return_idx,
    )

    g = torch.Generator()
    g.manual_seed(seed)

    print("prepare public dataloader")
    try:
        public_inv_dataloader = torch.utils.data.DataLoader(
            public_inv_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            worker_init_fn=worker_init_fn,
            generator=g,
        )
    except:
        public_inv_dataloader = torch.utils.data.DataLoader(
            public_inv_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            worker_init_fn=worker_init_fn,
        )

    return public_inv_dataloader


def prepare_facescrub_dataloaders(
    data_folder="/content",
    client_num=2,
    channel=1,
    batch_size=1,
    seed=42,
    num_workers=2,
    height=64,
    width=64,
    num_classes=530,
    crop=True,
    target_celeblities_num=100,
    blur_strength=10,
):
    np_resized_imgs = np.load(f"{data_folder}/resized_faces.npy")
    np_resized_labels = np.load(f"{data_folder}/resized_labels.npy")

    res = np.unique(np_resized_labels, return_counts=True)
    name2id = {name: i for i, name in enumerate(res[0])}
    id2name = {v: k for k, v in name2id.items()}

    local_identities_names = np.array_split(
        random.sample(res[0].tolist(), target_celeblities_num),
        client_num,
    )
    local_identities = [
        [name2id[name] for name in name_list] for name_list in local_identities_names
    ]

    name_id2client_id = {}
    for client_id, name_id_list in enumerate(local_identities):
        for idx in name_id_list:
            name_id2client_id[idx] = client_id

    X_public_list = []
    y_public_list = []
    is_sensitive_public_list = []
    X_private_lists = [[] for _ in range(client_num)]
    y_private_lists = [[] for _ in range(client_num)]

    for i in range(res[0].shape[0]):
        if i in name_id2client_id:
            idx = np.where(np_resized_labels == id2name[i])[0]
            sep_idxs = np.array_split(idx, 2)
            temp_array = []
            for temp_idx in range(np_resized_imgs[sep_idxs[0]].shape[0]):
                temp_array.append(
                    cv2.blur(
                        np_resized_imgs[sep_idxs[0]][temp_idx],
                        (blur_strength, blur_strength),
                    )
                )
            X_public_list.append(np.stack(temp_array))
            y_public_list += [i for _ in range(len(sep_idxs[0]))]
            is_sensitive_public_list += [0 for _ in range(len(sep_idxs[0]))]
            X_private_lists[name_id2client_id[i]].append(np_resized_imgs[sep_idxs[0]])
            y_private_lists[name_id2client_id[i]] += [
                i for _ in range(len(sep_idxs[0]))
            ]
        else:
            idx = np.where(np_resized_labels == id2name[i])[0]
            sep_idxs = np.array_split(idx, 2)

            X_public_list.append(np_resized_imgs[sep_idxs[0]])
            y_public_list += [i for _ in range(len(sep_idxs[0]))]
            is_sensitive_public_list += [1 for _ in range(len(sep_idxs[0]))]

            temp_array = []
            for temp_idx in range(np_resized_imgs[sep_idxs[1]].shape[0]):
                temp_array.append(
                    cv2.blur(
                        np_resized_imgs[sep_idxs[1]][temp_idx],
                        (blur_strength, blur_strength),
                    )
                )
            X_public_list.append(np.stack(temp_array))
            y_public_list += [i for _ in range(len(sep_idxs[1]))]
            is_sensitive_public_list += [0 for _ in range(len(sep_idxs[1]))]

    X_public = np.concatenate(X_public_list)
    y_public = np.array(y_public_list)
    is_sensitive_public = np.array(is_sensitive_public_list)

    (
        X_public_train,
        _,
        y_public_train,
        _,
        is_sensitive_public_train,
        _,
    ) = train_test_split(
        X_public,
        y_public,
        is_sensitive_public,
        test_size=0.1,
        random_state=42,
        stratify=y_public_list,
    )

    # X_test = np.concatenate(X_private_test_list + [X_public_test], axis=0)
    # y_test = np.concatenate(y_private_test_list + [y_public_test], axis=0)

    transforms_list = [transforms.ToTensor()]
    if channel == 1:
        transforms_list.append(transforms.Grayscale())
    if crop:
        transforms_list.append(transforms.CenterCrop((max(height, width))))
    else:
        transforms_list.append(transforms.Resize((height, width)))
    if channel == 1:
        transforms_list.append(transforms.Normalize((0.5,), (0.5,)))
    else:
        transforms_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    transform = transforms.Compose(transforms_list)
    return_idx = True

    sensitive_idx = np.where(is_sensitive_public_train == 1)[0]
    nonsensitive_idx = np.where(is_sensitive_public_train == 0)[0]
    X_public_input_inv = []
    X_public_output_inv = []

    skipped = 0
    for y in list(np.unique(y_public_train)):
        y_idx = np.where(y_public_train == y)[0]
        y_sensitive_idx = list(set(list(y_idx)) & set(list(sensitive_idx)))
        y_nonsensitive_idx = list(set(list(y_idx)) & set(list(nonsensitive_idx)))

        if len(y_sensitive_idx) != 0:
            blurred = []
            for idx in y_sensitive_idx:
                blurred.append(
                    cv2.blur(
                        X_public_train[idx],
                        (blur_strength, blur_strength),
                    )
                )

            X_public_input_inv.append(X_public_train[y_sensitive_idx])
            X_public_output_inv.append(np.stack(blurred))
        else:
            skipped += 1

    print("#skipped", skipped)

    X_public_input_inv = np.concatenate(X_public_input_inv)
    X_public_output_inv = np.concatenate(X_public_output_inv)

    public_inv_dataset = NumpyAEDataset(
        x=X_public_input_inv,
        y=X_public_output_inv,
        transform=transform,
        return_idx=return_idx,
    )

    g = torch.Generator()
    g.manual_seed(seed)

    print("prepare public dataloader")
    try:
        public_inv_dataloader = torch.utils.data.DataLoader(
            public_inv_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            worker_init_fn=worker_init_fn,
            generator=g,
        )
    except:
        public_inv_dataloader = torch.utils.data.DataLoader(
            public_inv_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            worker_init_fn=worker_init_fn,
        )

    return public_inv_dataloader


def prepare_inv_dataloaders(dataset_name, *args, **kwargs):
    """Return dataloaders

    Args:
        dataset_name (str): name of dataset (`LAG` or `LFW`)

    Raises:
        NotImplementedError: if name is not LAG or LFW.

    Returns:
        a tuple of public dataloader, a list of local dataloaders, test dataloader,
        and the list of target labels.
    """
    if dataset_name == "LAG":
        return prepare_inv_lag_dataloaders(*args, **kwargs)
    elif dataset_name == "LFW":
        return prepare_inv_lfw_dataloaders(*args, **kwargs)
    elif dataset_name == "FaceScrub":
        return prepare_facescrub_dataloaders(*args, **kwargs)
    else:
        raise NotImplementedError(f"{dataset_name} is not supported")
