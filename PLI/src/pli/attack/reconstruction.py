import math
import os

import numpy as np
import torch


def reconstruct_all_possible_targets(
    attack_type,
    local_identities,
    inv,
    output_dim,
    id2label,
    client_num,
    output_dir,
    device,
    base_name="",
):
    inv.eval()
    target_ids = sum(local_identities, [])

    target_labels = [id2label[celeb_id] for celeb_id in target_ids]
    target_labels_batch = np.array_split(
        target_labels, math.ceil(len(target_labels) / 64)
    )

    for label_batch in target_labels_batch:
        label_batch_tensor = torch.eye(output_dim)[label_batch].to(device)
        xs_rec = inv(label_batch_tensor.reshape(len(label_batch), -1, 1, 1))
        xs_rec_array = xs_rec.detach().cpu().numpy()

        for i, label in enumerate(label_batch):
            np.save(
                os.path.join(
                    output_dir,
                    f"{base_name}_{label}_{attack_type}",
                ),
                xs_rec_array[i],
            )

    return None


def reconstruct_all_possible_targets_with_pair_logits(
    attack_type,
    local_identities,
    inv,
    output_dim,
    id2label,
    output_dir,
    device,
    pi,
    pj,
    base_name="",
):
    inv.eval()
    target_ids = sum(local_identities, [])

    target_labels = [id2label[celeb_id] for celeb_id in target_ids]
    target_labels_batch = np.array_split(
        target_labels, math.ceil(len(target_labels) / 64)
    )

    for label_batch in target_labels_batch:
        dummy_pred_server = torch.ones(label_batch.shape[0], output_dim).to(device) * pi
        dummy_pred_server[:, label_batch] = pj
        dummy_pred_local = torch.eye(output_dim)[label_batch].to(device)
        dummy_preds = torch.cat([dummy_pred_server, dummy_pred_local], dim=1).to(device)

        xs_rec = inv(dummy_preds.reshape(len(label_batch), -1, 1, 1))
        xs_rec_array = xs_rec.detach().cpu().numpy()

        for i, label in enumerate(label_batch):
            np.save(
                os.path.join(
                    output_dir,
                    f"{base_name}_{label}_{attack_type}",
                ),
                xs_rec_array[i],
            )

    return None
