import glob
import math
import os
import random

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split

from ..utilsdata.utils import NumpyDataset, worker_init_fn


def prepare_lfw_dataloaders(
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
    X_private_list = [np.stack(x) for x in X_private_lists]
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

    public_dataset = NumpyDataset(
        x=X_public,
        y=y_public,
        transform=transform,
        return_idx=return_idx,
    )
    private_dataset_list = [
        NumpyDataset(
            x=X_private_list[i],
            y=y_private_list[i],
            transform=transform,
            return_idx=return_idx,
        )
        for i in range(client_num)
    ]

    g = torch.Generator()
    g.manual_seed(seed)

    print("prepare public dataloader")
    try:
        public_dataloader = torch.utils.data.DataLoader(
            public_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            worker_init_fn=worker_init_fn,
            generator=g,
        )
    except:
        public_dataloader = torch.utils.data.DataLoader(
            public_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            worker_init_fn=worker_init_fn,
        )

    print("prepare local dataloader")
    try:
        local_dataloaders = [
            torch.utils.data.DataLoader(
                private_dataset_list[i],
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                worker_init_fn=worker_init_fn,
                generator=g,
            )
            for i in range(client_num)
        ]
    except:
        local_dataloaders = [
            torch.utils.data.DataLoader(
                private_dataset_list[i],
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                worker_init_fn=worker_init_fn,
            )
            for i in range(client_num)
        ]

    return (
        public_dataloader,
        local_dataloaders,
        None,
        local_identities,
        is_sensitive_public,
    )


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

    res = list(np.unique(np_resized_labels, return_counts=True))

    if num_classes != 530:
        res[0] = res[0][np.argsort(-res[1])[:num_classes]]
        slice_idx = [i for i, name in enumerate(np_resized_labels) if name in res[0]]
        np_resized_imgs = np_resized_imgs[slice_idx]
        np_resized_labels = np_resized_labels[slice_idx]

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
            X_private_lists[name_id2client_id[i]].append(np_resized_imgs[sep_idxs[1]])
            y_private_lists[name_id2client_id[i]] += [
                i for _ in range(len(sep_idxs[1]))
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
    X_private_list = [np.concatenate(x) for x in X_private_lists]
    y_private_list = [np.array(y) for y in y_private_lists]

    (
        X_public_train,
        X_public_test,
        y_public_train,
        y_public_test,
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

    X_private_train_list = []
    X_private_test_list = []
    y_private_train_list = []
    y_private_test_list = []

    for X_private, y_private in zip(X_private_list, y_private_list):
        (
            X_private_train,
            X_private_test,
            y_private_train,
            y_private_test,
        ) = train_test_split(
            X_private, y_private, test_size=0.1, random_state=42, stratify=y_private
        )
        X_private_train_list.append(X_private_train)
        X_private_test_list.append(X_private_test)
        y_private_train_list.append(y_private_train)
        y_private_test_list.append(y_private_test)

    X_test = np.concatenate(X_private_test_list + [X_public_test], axis=0)
    y_test = np.concatenate(y_private_test_list + [y_public_test], axis=0)

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

    public_dataset_train = NumpyDataset(
        x=X_public_train,
        y=y_public_train,
        transform=transform,
        return_idx=return_idx,
    )

    private_dataset_train_list = [
        NumpyDataset(
            x=X_private_train_list[i],
            y=y_private_train_list[i],
            transform=transform,
            return_idx=return_idx,
        )
        for i in range(client_num)
    ]

    test_dataset = NumpyDataset(
        x=X_test,
        y=y_test,
        transform=transform,
        return_idx=return_idx,
    )

    g = torch.Generator()
    g.manual_seed(seed)

    print("prepare public dataloader")
    try:
        public_train_dataloader = torch.utils.data.DataLoader(
            public_dataset_train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            worker_init_fn=worker_init_fn,
            generator=g,
        )
    except:
        public_train_dataloader = torch.utils.data.DataLoader(
            public_dataset_train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            worker_init_fn=worker_init_fn,
        )

    print("prepare local dataloader")
    try:
        local_train_dataloaders = [
            torch.utils.data.DataLoader(
                private_dataset_train_list[i],
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                worker_init_fn=worker_init_fn,
                generator=g,
            )
            for i in range(client_num)
        ]
    except:
        local_train_dataloaders = [
            torch.utils.data.DataLoader(
                private_dataset_train_list[i],
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                worker_init_fn=worker_init_fn,
            )
            for i in range(client_num)
        ]

    try:
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            worker_init_fn=worker_init_fn,
            generator=g,
        )
    except:
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            worker_init_fn=worker_init_fn,
        )

    print("sensitive labels are", sorted(sum(local_identities, [])))

    return (
        public_train_dataloader,
        local_train_dataloaders,
        test_dataloader,
        local_identities,
        is_sensitive_public_train,
    )


def prepare_lag_dataloaders(
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

    local_identities_names = random.sample(
        list(unique_name_list), target_celeblities_num
    )
    local_identities_names = np.array_split(local_identities_names, client_num)
    local_identities_names = [id_list.tolist() for id_list in local_identities_names]
    local_identities = [
        [name2id[name] for name in name_list] for name_list in local_identities_names
    ]

    alloc = [-1 for _ in range(df.shape[0])]
    for j, (ay, name) in enumerate(zip(df["ay"].tolist(), df["name"].tolist())):
        if ay == 1:
            for i in range(client_num):
                if name in local_identities_names[i]:
                    alloc[j] = i + 1
                    break
        if alloc[j] == -1:
            alloc[j] = 0
    df["alloc"] = alloc

    X_public = np.stack([cv2.imread(p) for p in df[df["alloc"] == 0]["path"].tolist()])
    is_sensitive_public = df[df["alloc"] == 0]["ay"].values
    y_public = np.array([name2id[n] for n in df[df["alloc"] == 0]["name"].tolist()])
    X_private_list = [
        np.stack([cv2.imread(p) for p in df[df["alloc"] == i + 1]["path"].tolist()])
        for i in range(client_num)
    ]
    y_private_list = [
        np.array([name2id[n] for n in df[df["alloc"] == i + 1]["name"].tolist()])
        for i in range(client_num)
    ]

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

    public_dataset = NumpyDataset(
        x=X_public,
        y=y_public,
        transform=transform,
        return_idx=return_idx,
    )
    private_dataset_list = [
        NumpyDataset(
            x=X_private_list[i],
            y=y_private_list[i],
            transform=transform,
            return_idx=return_idx,
        )
        for i in range(client_num)
    ]

    g = torch.Generator()
    g.manual_seed(seed)

    print("prepare public dataloader")
    try:
        public_dataloader = torch.utils.data.DataLoader(
            public_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            worker_init_fn=worker_init_fn,
            generator=g,
        )
    except:
        public_dataloader = torch.utils.data.DataLoader(
            public_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            worker_init_fn=worker_init_fn,
        )

    print("prepare local dataloader")
    try:
        local_dataloaders = [
            torch.utils.data.DataLoader(
                private_dataset_list[i],
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                worker_init_fn=worker_init_fn,
                generator=g,
            )
            for i in range(client_num)
        ]
    except:
        local_dataloaders = [
            torch.utils.data.DataLoader(
                private_dataset_list[i],
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                worker_init_fn=worker_init_fn,
            )
            for i in range(client_num)
        ]

    return (
        public_dataloader,
        local_dataloaders,
        None,
        local_identities,
        is_sensitive_public,
    )


def extract_face(img, eye_left_x, eye_left_y, eye_right_x, eye_right_y, rate=0.65):
    """Implementation of `Contributions to facial feature extraction for Face recognition
    5: Face cropping based on eyes coordinates scheme.`"""
    W = img.shape[0]
    H = img.shape[1]
    xc = W / 2
    yc = H / 2
    x1, y1 = eye_left_x, eye_left_y
    x2, y2 = eye_right_x, eye_right_y
    phi = math.atan2(y2 - y1, x2 - x1)
    M = np.array([[math.cos(phi), -math.sin(phi)], [math.sin(phi), math.cos(phi)]])

    r1 = np.array([[xc, yc]]) + np.array([x1 - xc, y1 - yc]).T @ M
    r2 = np.array([[xc, yc]]) + np.array([x2 - xc, y2 - yc]).T @ M
    x1r, y1r = r1[0, 0], r1[0, 1]
    x2r, y2r = r2[0, 0], r2[0, 1]

    dist = np.sqrt((x1r - x2r) ** 2 + (y1r - y2r) ** 2)
    xul, yul = x1r - rate * dist, y1r - rate * dist
    xlr, ylr = x2r + rate * dist, y2r + (1 + rate) * dist

    return img[max(0, int(yul)) : max(0, int(ylr)), max(0, int(xul)) : max(0, int(xlr))]


def prepare_celeba_dataloaders(
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
    with open("materials/identity_CelebA.txt") as f:
        identities = f.readlines()

    label2path = {}
    path2label = {}
    for line in identities:
        path = line.split(" ")[0]
        label = int(line.split(" ")[1].split("\n")[0])
        if label not in label2path:
            label2path[label] = [path]
        else:
            label2path[label].append(path)
        path2label[path] = label

    df = pd.read_csv("materials/list_attr_celeba.csv")
    df["identity"] = df["image_id"].apply(lambda x: path2label[x])
    df_land = pd.read_csv("materials/list_landmarks_align_celeba.csv")

    target_celeblities = (
        df.groupby("identity")
        .count()["image_id"]
        .sort_values()
        .index.values[-1 * num_classes :]
    )

    name2id = {name: i for i, name in enumerate(target_celeblities)}
    local_identities_names = np.array_split(
        random.sample(target_celeblities.tolist(), target_celeblities_num),
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

    for name in target_celeblities:
        if name2id[name] in name_id2client_id:
            image_ids = df[df["identity"] == name]["image_id"].values
            sep_idxs = np.array_split(image_ids, 2)
            temp_array = []
            for temp_idx in sep_idxs[0]:
                img = cv2.imread(os.path.join(data_folder, temp_idx))
                image_land = df_land[df["image_id"] == temp_idx]
                img = extract_face(
                    img,
                    image_land["lefteye_x"],
                    image_land["lefteye_y"],
                    image_land["righteye_x"],
                    image_land["righteye_y"],
                )
                temp_array.append(
                    cv2.blur(
                        cv2.resize(
                            img,
                            dsize=(width, height),
                        ),
                        (blur_strength, blur_strength),
                    )
                )
            X_public_list.append(np.stack(temp_array))
            y_public_list += [name2id[name] for _ in range(len(sep_idxs[0]))]
            is_sensitive_public_list += [0 for _ in range(len(sep_idxs[0]))]

            temp_array = []
            for temp_idx in sep_idxs[1]:
                img = cv2.imread(os.path.join(data_folder, temp_idx))
                image_land = df_land[df["image_id"] == temp_idx]
                img = extract_face(
                    img,
                    image_land["lefteye_x"],
                    image_land["lefteye_y"],
                    image_land["righteye_x"],
                    image_land["righteye_y"],
                )
                temp_array.append(
                    cv2.resize(
                        img,
                        dsize=(width, height),
                    )
                )
            X_private_lists[name_id2client_id[name2id[name]]].append(
                np.stack(temp_array)
            )
            y_private_lists[name_id2client_id[name2id[name]]] += [
                name2id[name] for _ in range(len(sep_idxs[1]))
            ]

        else:
            image_ids = df[df["identity"] == name]["image_id"].values
            sep_idxs = np.array_split(image_ids, 2)
            temp_array = []
            for temp_idx in sep_idxs[0]:
                img = cv2.imread(os.path.join(data_folder, temp_idx))
                image_land = df_land[df["image_id"] == temp_idx]
                img = extract_face(
                    img,
                    image_land["lefteye_x"],
                    image_land["lefteye_y"],
                    image_land["righteye_x"],
                    image_land["righteye_y"],
                )
                temp_array.append(
                    cv2.blur(
                        cv2.resize(
                            img,
                            dsize=(width, height),
                        ),
                        (blur_strength, blur_strength),
                    )
                )
            X_public_list.append(np.stack(temp_array))
            y_public_list += [name2id[name] for _ in range(len(sep_idxs[0]))]
            is_sensitive_public_list += [0 for _ in range(len(sep_idxs[0]))]

            temp_array = []
            for temp_idx in sep_idxs[1]:
                img = cv2.imread(os.path.join(data_folder, temp_idx))
                image_land = df_land[df["image_id"] == temp_idx]
                img = extract_face(
                    img,
                    image_land["lefteye_x"],
                    image_land["lefteye_y"],
                    image_land["righteye_x"],
                    image_land["righteye_y"],
                )
                temp_array.append(
                    cv2.resize(
                        img,
                        dsize=(width, height),
                    )
                )
            X_public_list.append(np.stack(temp_array))
            y_public_list += [name2id[name] for _ in range(len(sep_idxs[1]))]
            is_sensitive_public_list += [1 for _ in range(len(sep_idxs[1]))]

    X_public = np.concatenate(X_public_list)
    y_public = np.array(y_public_list)
    is_sensitive_public = np.array(is_sensitive_public_list)
    X_private_list = [np.concatenate(x) for x in X_private_lists]
    y_private_list = [np.array(y) for y in y_private_lists]

    (
        X_public_train,
        X_public_test,
        y_public_train,
        y_public_test,
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

    X_private_train_list = []
    X_private_test_list = []
    y_private_train_list = []
    y_private_test_list = []

    for X_private, y_private in zip(X_private_list, y_private_list):
        (
            X_private_train,
            X_private_test,
            y_private_train,
            y_private_test,
        ) = train_test_split(
            X_private, y_private, test_size=0.1, random_state=42, stratify=y_private
        )
        X_private_train_list.append(X_private_train)
        X_private_test_list.append(X_private_test)
        y_private_train_list.append(y_private_train)
        y_private_test_list.append(y_private_test)

    X_test = np.concatenate(X_private_test_list + [X_public_test], axis=0)
    y_test = np.concatenate(y_private_test_list + [y_public_test], axis=0)

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

    public_dataset_train = NumpyDataset(
        x=X_public_train,
        y=y_public_train,
        transform=transform,
        return_idx=return_idx,
    )

    private_dataset_train_list = [
        NumpyDataset(
            x=X_private_train_list[i],
            y=y_private_train_list[i],
            transform=transform,
            return_idx=return_idx,
        )
        for i in range(client_num)
    ]

    test_dataset = NumpyDataset(
        x=X_test,
        y=y_test,
        transform=transform,
        return_idx=return_idx,
    )

    g = torch.Generator()
    g.manual_seed(seed)

    print("prepare public dataloader")
    try:
        public_train_dataloader = torch.utils.data.DataLoader(
            public_dataset_train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            worker_init_fn=worker_init_fn,
            generator=g,
        )
    except:
        public_train_dataloader = torch.utils.data.DataLoader(
            public_dataset_train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            worker_init_fn=worker_init_fn,
        )

    print("prepare local dataloader")
    try:
        local_train_dataloaders = [
            torch.utils.data.DataLoader(
                private_dataset_train_list[i],
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                worker_init_fn=worker_init_fn,
                generator=g,
            )
            for i in range(client_num)
        ]
    except:
        local_train_dataloaders = [
            torch.utils.data.DataLoader(
                private_dataset_train_list[i],
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                worker_init_fn=worker_init_fn,
            )
            for i in range(client_num)
        ]

    try:
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            worker_init_fn=worker_init_fn,
            generator=g,
        )
    except:
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            worker_init_fn=worker_init_fn,
        )

    print("sensitive labels are", sorted(sum(local_identities, [])))

    return (
        public_train_dataloader,
        local_train_dataloaders,
        test_dataloader,
        local_identities,
        is_sensitive_public_train,
    )


def prepare_dataloaders(dataset_name, *args, **kwargs):
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
        return prepare_lag_dataloaders(*args, **kwargs)
    elif dataset_name == "LFW":
        return prepare_lfw_dataloaders(*args, **kwargs)
    elif dataset_name == "FaceScrub":
        return prepare_facescrub_dataloaders(*args, **kwargs)
    elif dataset_name == "CelebA":
        return prepare_celeba_dataloaders(*args, **kwargs)
    else:
        raise NotImplementedError(f"{dataset_name} is not supported")
