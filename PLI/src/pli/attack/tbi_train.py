import math
import os

import numpy as np
import torch

from ..utils.tbi_setup import (
    setup_our_inv_dataloader,
    setup_our_inv_dataloader_from_single_client,
    setup_tbi_inv_dataloader,
)
from .confidence import get_pi, get_pj
from .reconstruction import (
    reconstruct_all_possible_targets,
    reconstruct_all_possible_targets_with_pair_logits,
)


def train_tbi_inv_model(data, device, inv_model, optimizer, criterion):
    x = data[0].to(device)
    y_pred_local = data[1].to(device)

    optimizer.zero_grad()
    x_rec_original = inv_model(y_pred_local.reshape(x.shape[0], -1, 1, 1))
    loss = criterion(x, x_rec_original)
    loss.backward()
    optimizer.step()

    return loss, x, x_rec_original


def train_our_inv_model(
    data, prior, device, inv_model, optimizer, criterion, gamma=0.1
):
    x = data[0].to(device)
    y_pred_local = data[2].to(device)
    y_label = data[3]

    optimizer.zero_grad()
    x_rec_original = inv_model(y_pred_local.reshape(x.shape[0], -1, 1, 1))
    if gamma != 0:
        loss = criterion(x, x_rec_original) + gamma * criterion(
            prior[y_label].to(device), x_rec_original
        )
    else:
        loss = criterion(x, x_rec_original)
    loss.backward()
    optimizer.step()

    return loss, x, x_rec_original


def train_our_inv_model_with_pair_logits(
    data,
    prior,
    device,
    inv_model,
    optimizer,
    criterion,
    gamma=0.1,
):
    x = data[0].to(device)
    y_pred_server = data[1].to(device)
    y_pred_local = data[2].to(device)
    y_label = data[3]

    y_preds_server_and_local = torch.cat([y_pred_server, y_pred_local], dim=1)

    optimizer.zero_grad()
    x_rec_original = inv_model(y_preds_server_and_local.reshape(x.shape[0], -1, 1, 1))
    if gamma != 0:
        loss = criterion(x, x_rec_original) + gamma * criterion(
            prior[y_label].to(device), x_rec_original
        )
    else:
        loss = criterion(x, x_rec_original)
    loss.backward()
    optimizer.step()

    return loss, x, x_rec_original


def train_our_inv_model_with_only_priors(
    target_labels, prior, device, inv_model, optimizer, criterion, gamma=0.1
):
    running_loss = 0

    if gamma != 0:
        output_dim = prior.shape[0]
        target_labels_batch = np.array_split(
            target_labels, math.ceil(len(target_labels) / 64)
        )
        for label_batch in target_labels_batch:
            optimizer.zero_grad()
            label_batch_tensor = torch.eye(output_dim)[label_batch].to(device)
            xs_rec = inv_model(label_batch_tensor.reshape(len(label_batch), -1, 1, 1))
            loss = gamma * criterion(prior[label_batch], xs_rec.cpu())
            loss.backward()
            optimizer.step()

            running_loss += loss.item() / len(target_labels_batch)

    return running_loss


def train_our_inv_model_with_only_priors_paird_logits(
    target_labels, prior, device, inv_model, optimizer, criterion, pi, pj, gamma=0.1
):
    running_loss = 0

    if gamma != 0:
        output_dim = prior.shape[0]
        target_labels_batch = np.array_split(
            target_labels, math.ceil(len(target_labels) / 64)
        )
        for label_batch in target_labels_batch:
            optimizer.zero_grad()
            dummy_pred_server = (
                torch.ones(label_batch.shape[0], output_dim).to(device) * pi
            )
            dummy_pred_server[:, label_batch] = pj
            dummy_pred_local = torch.eye(output_dim)[label_batch].to(device)
            dummy_preds = torch.cat([dummy_pred_server, dummy_pred_local], dim=1).to(
                device
            )
            xs_rec = inv_model(dummy_preds.reshape(len(label_batch), -1, 1, 1))
            loss = gamma * criterion(prior[label_batch], xs_rec.cpu())
            loss.backward()
            optimizer.step()

            running_loss += loss.item() / len(target_labels_batch)

    return running_loss


def train_our_inv_model_on_logits_dataloader(
    prediction_dataloader,
    prior,
    device,
    inv,
    inv_optimizer,
    criterion,
    ablation_study,
    gamma=0.1,
):
    inv_running_loss = 0
    running_size = 0
    inv.train()
    for data in prediction_dataloader:
        if ablation_study != 2:
            loss, x, x_rec = train_our_inv_model(
                data, prior, device, inv, inv_optimizer, criterion, gamma=gamma
            )
        else:
            loss, x, x_rec = train_our_inv_model_with_pair_logits(
                data, prior, device, inv, inv_optimizer, criterion, gamma=gamma
            )
        inv_running_loss += loss.item()
        running_size += x.shape[0]

    return inv_running_loss / running_size, x, x_rec


def get_our_inv_train_func(
    client_num,
    is_sensitive_flag,
    local_identities,
    inv_transform,
    return_idx,
    seed,
    batch_size,
    num_workers,
    device,
    inv_tempreature,
    inv_batch_size,
    inv_epoch,
    inv,
    inv_optimizer,
    prior,
    criterion,
    output_dim,
    attack_type,
    id2label,
    output_dir,
    ablation_study,
    alpha,
    gamma=0.1,
    only_sensitive=True,
):
    def inv_train(api):
        target_client_apis = [
            lambda x_: api.clients[target_client_id](x_).detach()
            for target_client_id in range(client_num)
        ]

        # --- Prepare Public Dataset --- #
        # target_labels = local_identities[target_client_id]

        target_labels = sum(
            [[id2label[la] for la in temp_list] for temp_list in local_identities], []
        )

        prediction_dataloader = setup_our_inv_dataloader(
            target_labels,
            is_sensitive_flag,
            api,
            target_client_apis,
            inv_transform,
            return_idx,
            seed,
            batch_size,
            num_workers,
            device,
            inv_tempreature,
            inv_batch_size,
            only_sensitive,
        )

        # checkpoint = torch.load(inv_path_list[target_client_id] + ".pth")
        # inv.load_state_dict(checkpoint["model"])
        # inv_optimizer.load_state_dict(checkpoint["optimizer"])

        for i in range(1, inv_epoch + 1):
            (inv_running_loss, _, _) = train_our_inv_model_on_logits_dataloader(
                prediction_dataloader,
                prior,
                device,
                inv,
                inv_optimizer,
                criterion,
                ablation_study,
                gamma=gamma,
            )

            print(f"inv epoch={i}, inv loss ", inv_running_loss)

            if ablation_study == 0:
                inv_prior_loss = train_our_inv_model_with_only_priors(
                    target_labels,
                    prior,
                    device,
                    inv,
                    inv_optimizer,
                    criterion,
                    gamma=gamma,
                )

                print(f"inv epoch={i}, prior loss ", inv_prior_loss)
            elif ablation_study == 2:
                pi = get_pi(output_dim, alpha)
                pj = get_pj(output_dim, alpha)
                inv_prior_loss = train_our_inv_model_with_only_priors_paird_logits(
                    target_labels,
                    prior,
                    device,
                    inv,
                    inv_optimizer,
                    criterion,
                    pi,
                    pj,
                    gamma=gamma,
                )

                print(f"inv epoch={i}, prior loss ", inv_prior_loss)

            with open(
                os.path.join(output_dir, "inv_result.txt"),
                "a",
                encoding="utf-8",
                newline="\n",
            ) as f:
                f.write(f"{i}, {inv_running_loss}\n")

        # state = {
        #    "model": inv.state_dict(),
        #    "optimizer": inv_optimizer.state_dict(),
        # }
        # torch.save(state, inv_path_list[target_client_id] + ".pth")

        if api.epoch % 2 == 1:
            print("saving ...")
            if ablation_study != 2:
                reconstruct_all_possible_targets(
                    attack_type,
                    local_identities,
                    inv,
                    output_dim,
                    id2label,
                    client_num,
                    output_dir,
                    device,
                    base_name=api.epoch,
                )
            else:
                pi = get_pi(output_dim, alpha)
                pj = get_pj(output_dim, alpha)
                reconstruct_all_possible_targets_with_pair_logits(
                    attack_type,
                    local_identities,
                    inv,
                    output_dim,
                    id2label,
                    output_dir,
                    device,
                    pi,
                    pj,
                    base_name=api.epoch,
                )

    return inv_train


def get_our_inv_train_func_with_multi_models(
    client_num,
    is_sensitive_flag,
    local_identities,
    inv_transform,
    return_idx,
    seed,
    batch_size,
    num_workers,
    device,
    inv_path_list,
    inv_tempreature,
    inv_batch_size,
    inv_epoch,
    inv,
    inv_optimizer,
    prior,
    criterion,
    output_dim,
    attack_type,
    id2label,
    output_dir,
    ablation_study,
    alpha,
    gamma=0.1,
    only_sensitive=True,
):
    def inv_train(api):
        target_labels = sum(
            [[id2label[la] for la in temp_list] for temp_list in local_identities], []
        )

        for target_client_id in range(client_num):

            def target_client_api(x_):
                return api.clients[target_client_id](x_).detach()

            prediction_dataloader = setup_our_inv_dataloader_from_single_client(
                target_labels,
                is_sensitive_flag,
                api,
                target_client_api,
                inv_transform,
                return_idx,
                seed,
                batch_size,
                num_workers,
                device,
                inv_tempreature,
                inv_batch_size,
                only_sensitive,
            )

            checkpoint = torch.load(inv_path_list[target_client_id] + ".pth")
            inv.load_state_dict(checkpoint["model"])
            inv_optimizer.load_state_dict(checkpoint["optimizer"])

            for i in range(1, inv_epoch + 1):
                (inv_running_loss, _, _) = train_our_inv_model_on_logits_dataloader(
                    prediction_dataloader,
                    prior,
                    device,
                    inv,
                    inv_optimizer,
                    criterion,
                    ablation_study,
                    gamma=gamma,
                )

                print(f"inv epoch={i}, inv loss ", inv_running_loss)

                if ablation_study == 0:
                    inv_prior_loss = train_our_inv_model_with_only_priors(
                        target_labels,
                        prior,
                        device,
                        inv,
                        inv_optimizer,
                        criterion,
                        gamma=gamma,
                    )

                    print(f"inv epoch={i}, prior loss ", inv_prior_loss)
                elif ablation_study == 2:
                    pi = get_pi(output_dim, alpha)
                    pj = get_pj(output_dim, alpha)
                    inv_prior_loss = train_our_inv_model_with_only_priors_paird_logits(
                        target_labels,
                        prior,
                        device,
                        inv,
                        inv_optimizer,
                        criterion,
                        pi,
                        pj,
                        gamma=gamma,
                    )

                    print(f"inv epoch={i}, prior loss ", inv_prior_loss)

                with open(
                    os.path.join(output_dir, "inv_result.txt"),
                    "a",
                    encoding="utf-8",
                    newline="\n",
                ) as f:
                    f.write(f"{i}, {inv_running_loss}\n")

            state = {
                "model": inv.state_dict(),
                "optimizer": inv_optimizer.state_dict(),
            }
            torch.save(state, inv_path_list[target_client_id] + ".pth")

            if api.epoch % 2 == 1:
                print("saving ...")
                if ablation_study != 2:
                    reconstruct_all_possible_targets(
                        attack_type,
                        local_identities,
                        inv,
                        output_dim,
                        id2label,
                        client_num,
                        output_dir,
                        device,
                        base_name=f"{api.epoch}_{target_client_id}",
                    )
                else:
                    pi = get_pi(output_dim, alpha)
                    pj = get_pj(output_dim, alpha)
                    reconstruct_all_possible_targets_with_pair_logits(
                        attack_type,
                        local_identities,
                        inv,
                        output_dim,
                        id2label,
                        output_dir,
                        device,
                        pi,
                        pj,
                        base_name=f"{api.epoch}_{target_client_id}",
                    )

    return inv_train


def get_tbi_inv_train_func(
    client_num,
    local_identities,
    inv_transform,
    return_idx,
    seed,
    batch_size,
    num_workers,
    device,
    inv_tempreature,
    inv_batch_size,
    inv_epoch,
    inv,
    inv_optimizer,
    criterion,
    output_dir,
    attack_type,
    output_dim,
    id2label,
):
    def inv_train(api):
        target_client_apis = [
            lambda x_: api.clients[target_client_id](x_).detach()
            for target_client_id in range(client_num)
        ]

        # --- Prepare Public Dataset --- #
        # target_labels = local_identities[target_client_id]
        target_labels = sum(local_identities, [])
        prediction_dataloader = setup_tbi_inv_dataloader(
            target_labels,
            None,
            api,
            target_client_apis,
            inv_transform,
            return_idx,
            seed,
            batch_size,
            num_workers,
            device,
            inv_tempreature,
            inv_batch_size,
        )

        # checkpoint = torch.load(inv_path_list[target_client_id] + ".pth")
        # inv.load_state_dict(checkpoint["model"])
        # inv_optimizer.load_state_dict(checkpoint["optimizer"])

        for i in range(1, inv_epoch + 1):
            tbi_running_loss = 0
            running_size = 0
            for data in prediction_dataloader:
                loss, x, _ = train_tbi_inv_model(
                    data,
                    device,
                    inv,
                    inv_optimizer,
                    criterion,
                )
                tbi_running_loss += loss.item()
                running_size += x.shape[0]

            tbi_running_loss /= running_size
            print(f"inv epoch={i}, inv loss ", tbi_running_loss)

            with open(
                os.path.join(output_dir, "inv_result.txt"),
                "a",
                encoding="utf-8",
                newline="\n",
            ) as f:
                f.write(f"{i}, {tbi_running_loss}\n")

        if api.epoch % 2 == 1:
            print("saving ...")
            reconstruct_all_possible_targets(
                attack_type,
                local_identities,
                inv,
                output_dim,
                id2label,
                client_num,
                output_dir,
                device,
                base_name=api.epoch,
            )

        # state = {"model": inv.state_dict(), "optimizer": inv_optimizer.state_dict()}
        # torch.save(state, inv_path_list[target_client_id] + ".pth")

    return inv_train
