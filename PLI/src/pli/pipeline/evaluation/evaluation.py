import glob
import os
import pickle

import cv2
import numpy as np
import torch
import tqdm
from skimage.metrics import structural_similarity

from ...utils.loss import SSIMLoss
from ...utils.utils_data import (
    extract_transformd_dataset_from_dataloader,
    total_variance_numpy_batch,
)


def evaluation_full(
    client_num,
    num_classes,
    public_dataloader,
    local_dataloaders,
    local_identities,
    id2label,
    attack_type,
    output_dir,
    epoch=5,
    device="cuda:0",
    save_gt=True,
):
    print("evaluating ...")

    ssim_list = {
        f"{attack_type}_ssim_private": [],
        f"{attack_type}_ssim_public": [],
    }
    mse_list = {
        f"{attack_type}_mse_private": [],
        f"{attack_type}_mse_public": [],
    }
    result = {
        f"{attack_type}_success": 0,
        f"{attack_type}_too_close_to_public": 0,
    }

    target_ids = sum(local_identities, [])

    (
        public_dataset_transformed,
        public_dataset_label,
    ) = extract_transformd_dataset_from_dataloader(public_dataloader, return_idx=True)

    private_dataset_transformed_list = []
    private_dataset_label_list = []
    for i in range(client_num):
        temp_dataset, temp_label = extract_transformd_dataset_from_dataloader(
            local_dataloaders[i], return_idx=True
        )
        private_dataset_transformed_list.append(temp_dataset)
        private_dataset_label_list.append(temp_label)
    private_dataset_transformed = torch.cat(private_dataset_transformed_list)
    private_dataset_label = torch.cat(private_dataset_label_list)

    ssim = SSIMLoss()
    print("eval num_classes is ", num_classes)
    for label in range(num_classes):
        np.save(
            os.path.join(output_dir, "private_" + str(label)),
            cv2.cvtColor(
                private_dataset_transformed[private_dataset_label == label]
                .mean(dim=0)
                .detach()
                .cpu()
                .numpy()
                .transpose(1, 2, 0)
                * 0.5
                + 0.5,
                cv2.COLOR_BGR2RGB,
            ),
        )

        np.save(
            os.path.join(output_dir, "public_" + str(label)),
            cv2.cvtColor(
                public_dataset_transformed[public_dataset_label == label]
                .mean(dim=0)
                .detach()
                .cpu()
                .numpy()
                .transpose(1, 2, 0)
                * 0.5
                + 0.5,
                cv2.COLOR_BGR2RGB,
            ),
        )

    for celeb_id in tqdm.tqdm(target_ids):
        label = id2label[celeb_id]

        """
        if save_gt:
            np.save(
                os.path.join(output_dir, "private_" + str(label)),
                cv2.cvtColor(
                    private_dataset_transformed[private_dataset_label == label]
                    .mean(dim=0)
                    .detach()
                    .cpu()
                    .numpy()
                    .transpose(1, 2, 0)
                    * 0.5
                    + 0.5,
                    cv2.COLOR_BGR2RGB,
                ),
            )

            np.save(
                os.path.join(output_dir, "public_" + str(label)),
                cv2.cvtColor(
                    public_dataset_transformed[public_dataset_label == label]
                    .mean(dim=0)
                    .detach()
                    .cpu()
                    .numpy()
                    .transpose(1, 2, 0)
                    * 0.5
                    + 0.5,
                    cv2.COLOR_BGR2RGB,
                ),
            )
        """

        temp_path = glob.glob(
            os.path.join(output_dir, str(epoch) + "_" + str(label) + "_*")
        )[0]

        best_img_tensor = torch.Tensor(np.load(temp_path)).to(device)
        best_img_tensor_batch = torch.stack(
            [best_img_tensor.clone() for _ in range(num_classes)]
        ).to(device)
        best_img_tensor_batch = best_img_tensor_batch * 0.5 + 0.5

        private_data = torch.stack(
            [
                private_dataset_transformed[private_dataset_label == i].mean(dim=0)
                for i in range(num_classes)
            ]
        ).to(device)
        private_data = private_data * 0.5 + 0.5

        public_data = torch.stack(
            [
                public_dataset_transformed[public_dataset_label == i].mean(dim=0)
                for i in range(num_classes)
            ]
        ).to(device)
        public_data = public_data * 0.5 + 0.5

        # evaluation on ssim

        ssim_private_list = np.zeros(num_classes)
        ssim_public_list = np.zeros(num_classes)

        for idxs in np.array_split(list(range(num_classes)), 5):
            ssim_private_list[idxs] = (
                ssim(best_img_tensor_batch[idxs], private_data[idxs], False)
                .detach()
                .cpu()
                .numpy()
            )
            ssim_public_list[idxs] = (
                ssim(best_img_tensor_batch[idxs], public_data[idxs], False)
                .detach()
                .cpu()
                .numpy()
            )

        best_label = np.nanargmax(
            ssim_private_list.tolist() + ssim_public_list.tolist()
        )
        ssim_private = ssim_private_list[label]
        ssim_public = ssim_public_list[label]

        print(f"best_label is {best_label}, target_label is {label}")

        result[f"{attack_type}_success"] += label == best_label
        result[f"{attack_type}_too_close_to_public"] += (
            label + num_classes == best_label
        )
        ssim_list[f"{attack_type}_ssim_private"].append(ssim_private)
        ssim_list[f"{attack_type}_ssim_public"].append(ssim_public)

        # evaluation on mse

        mse_list[f"{attack_type}_mse_private"].append(
            torch.nn.functional.mse_loss(best_img_tensor, private_data[label]).item()
        )
        mse_list[f"{attack_type}_mse_public"].append(
            torch.nn.functional.mse_loss(best_img_tensor, public_data[label]).item()
        )

    for k in ssim_list.keys():
        if len(ssim_list[k]) > 0:
            result[k + "_mean"] = np.mean(ssim_list[k])
            result[k + "_std"] = np.std(ssim_list[k])

    for k in mse_list.keys():
        if len(mse_list[k]) > 0:
            result[k + "_mean"] = np.mean(mse_list[k])
            result[k + "_std"] = np.std(mse_list[k])

    return result


def evaluation_full_multi_models(
    client_num,
    num_classes,
    public_dataloader,
    local_dataloaders,
    local_identities,
    id2label,
    attack_type,
    output_dir,
    epoch=5,
    device="cuda:0",
    beta=0.1,
    save_gt=True,
    label_transform=False,
):
    print("evaluating ...")

    ssim_list = {
        f"{attack_type}_ssim_private": [],
        f"{attack_type}_ssim_public": [],
    }
    mse_list = {
        f"{attack_type}_mse_private": [],
        f"{attack_type}_mse_public": [],
    }
    result = {
        f"{attack_type}_success": 0,
        f"{attack_type}_too_close_to_public": 0,
    }

    target_ids = sum(local_identities, [])

    (
        public_dataset_transformed,
        public_dataset_label,
    ) = extract_transformd_dataset_from_dataloader(public_dataloader, return_idx=True)

    private_dataset_transformed_list = []
    private_dataset_label_list = []
    for i in range(client_num):
        temp_dataset, temp_label = extract_transformd_dataset_from_dataloader(
            local_dataloaders[i], return_idx=True
        )
        private_dataset_transformed_list.append(temp_dataset)
        private_dataset_label_list.append(temp_label)
    private_dataset_transformed = torch.cat(private_dataset_transformed_list)
    private_dataset_label = torch.cat(private_dataset_label_list)

    ssim = SSIMLoss()

    ssim_true = []
    ssim_false = []

    if label_transform:
        label2id = {v: k for k, v in id2label.items()}
    else:
        label2id = list(range(num_classes))

    for client_id, ldl in enumerate(local_dataloaders):
        np.save(
            os.path.join(output_dir, "label_" + str(client_id)),
            np.unique(ldl.dataset.y),
        )

    for celeb_id in tqdm.tqdm(target_ids):
        label = id2label[celeb_id]

        reconstructed_imgs = np.stack(
            [
                np.load(p)
                for p in glob.glob(
                    os.path.join(output_dir, str(epoch) + "_*_" + str(label) + "_*")
                )
            ]
        )

        ssim_matrix = np.zeros((len(reconstructed_imgs), len(reconstructed_imgs)))
        tv_array = total_variance_numpy_batch(reconstructed_imgs)
        for i in range(len(reconstructed_imgs)):
            for j in range(len(reconstructed_imgs)):
                if i != j:
                    ssim_matrix[i][j] = structural_similarity(
                        reconstructed_imgs[i].transpose(1, 2, 0) * 0.5 + 0.5,
                        reconstructed_imgs[j].transpose(1, 2, 0) * 0.5 + 0.5,
                        multichannel=True,
                        data_range=1,
                    )

        np.save(
            os.path.join(output_dir, "ssim_" + str(label)),
            ssim_matrix.sum(axis=0) / (client_num - 1),
        )

        best_img = reconstructed_imgs[
            np.argmin(ssim_matrix.sum(axis=0) / (client_num - 1) + beta * tv_array)
        ]
        np.save(
            os.path.join(output_dir, "best_" + str(label)),
            cv2.cvtColor(
                best_img.transpose(1, 2, 0) * 0.5 + 0.5,
                cv2.COLOR_BGR2RGB,
            ),
        )
        best_img_tensor = torch.Tensor(best_img).to(device)

        if save_gt:
            np.save(
                os.path.join(output_dir, "private_" + str(label)),
                cv2.cvtColor(
                    private_dataset_transformed[private_dataset_label == label]
                    .mean(dim=0)
                    .detach()
                    .cpu()
                    .numpy()
                    .transpose(1, 2, 0)
                    * 0.5
                    + 0.5,
                    cv2.COLOR_BGR2RGB,
                ),
            )

            np.save(
                os.path.join(output_dir, "public_" + str(label)),
                cv2.cvtColor(
                    public_dataset_transformed[public_dataset_label == label]
                    .mean(dim=0)
                    .detach()
                    .cpu()
                    .numpy()
                    .transpose(1, 2, 0)
                    * 0.5
                    + 0.5,
                    cv2.COLOR_BGR2RGB,
                ),
            )
        # temp_path = glob.glob(
        #    os.path.join(output_dir, str(epoch) + "_" + str(label) + "_*")
        # )[0]

        best_img_tensor_batch = torch.stack(
            [best_img_tensor.clone() for _ in range(num_classes)]
        ).to(device)
        best_img_tensor_batch = best_img_tensor_batch * 0.5 + 0.5

        private_data = torch.stack(
            [
                private_dataset_transformed[private_dataset_label == i].mean(dim=0)
                for i in range(num_classes)
            ]
        ).to(device)
        private_data = private_data * 0.5 + 0.5

        public_data = torch.stack(
            [
                public_dataset_transformed[public_dataset_label == i].mean(dim=0)
                for i in range(num_classes)
            ]
        ).to(device)
        public_data = public_data * 0.5 + 0.5

        # evaluation on ssim

        ssim_private_list = np.zeros(num_classes)
        ssim_public_list = np.zeros(num_classes)

        for idxs in np.array_split(list(range(num_classes)), 5):
            ssim_private_list[idxs] = (
                ssim(best_img_tensor_batch[idxs], private_data[idxs], False)
                .detach()
                .cpu()
                .numpy()
            )
            ssim_public_list[idxs] = (
                ssim(best_img_tensor_batch[idxs], public_data[idxs], False)
                .detach()
                .cpu()
                .numpy()
            )

        best_label = np.nanargmax(
            ssim_private_list.tolist() + ssim_public_list.tolist()
        )
        # print(label, best_label, label2id[label])
        ssim_private = ssim_private_list[label2id[label]]
        ssim_public = ssim_public_list[label2id[label]]

        result[f"{attack_type}_success"] += label2id[label] == best_label
        result[f"{attack_type}_too_close_to_public"] += (
            label2id[label] + num_classes == best_label
        )
        ssim_list[f"{attack_type}_ssim_private"].append(ssim_private)
        ssim_list[f"{attack_type}_ssim_public"].append(ssim_public)

        # evaluation on mse

        mse_list[f"{attack_type}_mse_private"].append(
            torch.nn.functional.mse_loss(
                best_img_tensor, private_data[label2id[label]]
            ).item()
        )
        mse_list[f"{attack_type}_mse_public"].append(
            torch.nn.functional.mse_loss(
                best_img_tensor, public_data[label2id[label]]
            ).item()
        )

    for k in ssim_list.keys():
        if len(ssim_list[k]) > 0:
            result[k + "_mean"] = np.mean(ssim_list[k])
            result[k + "_std"] = np.std(ssim_list[k])

    for k in mse_list.keys():
        if len(mse_list[k]) > 0:
            result[k + "_mean"] = np.mean(mse_list[k])
            result[k + "_std"] = np.std(mse_list[k])

    return result
