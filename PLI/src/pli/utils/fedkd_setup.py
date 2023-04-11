import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ..fedmd import (
    DSFLAPI,
    DSFLClient,
    DSFLServer,
    FedGEMSAPI,
    FedGEMSClient,
    FedGEMSServer,
    FedMDAPI,
    FedMDClient,
    FedMDServer,
)


def setup_fedmd(
    model_class,
    public_dataloader,
    local_dataloaders,
    test_dataloader,
    client_num=2,
    channel=1,
    lr=0.01,
    device="cpu",
    criterion=nn.CrossEntropyLoss(),
    num_communication=10,
    consensus_epoch=1,
    revisit_epoch=1,
    transfer_epoch_public=1,
    transfer_epoch_private=1,
    server_training_epoch=1,
    use_server_model=True,
    weight_decay=0.01,
    input_dim=112 * 94,
    output_dim=20,
    round_decimal=None,
    custom_action=lambda x: x,
):
    # setup clients
    clients = [
        FedMDClient(
            model_class(input_dim=input_dim, output_dim=output_dim, channel=channel).to(
                device
            ),
            public_dataloader,
            output_dim=output_dim,
            round_decimal=round_decimal,
            user_id=i,
            device=device,
        ).to(device)
        for i in range(client_num)
    ]
    client_optimizers = [
        optim.Adam(client.parameters(), lr=lr, weight_decay=weight_decay)
        for client in clients
    ]

    # setup a server
    if use_server_model:
        server_model = model_class(
            input_dim=input_dim, output_dim=output_dim, channel=channel
        ).to(device)
        server_optimizer = optim.Adam(
            server_model.parameters(), lr=lr, weight_decay=weight_decay
        )
        server = FedMDServer(clients, server_model=server_model, device=device).to(
            device
        )
    else:
        server = FedMDServer(clients, device=device).to(device)
        server_optimizer = None

    # setup FedMD
    fedmd_api = FedMDAPI(
        server=server,
        clients=clients,
        public_dataloader=public_dataloader,
        local_dataloaders=local_dataloaders,
        validation_dataloader=test_dataloader,
        criterion=criterion,
        client_optimizers=client_optimizers,
        server_optimizer=server_optimizer,
        num_communication=num_communication,
        consensus_epoch=consensus_epoch,
        revisit_epoch=revisit_epoch,
        transfer_epoch_public=transfer_epoch_public,
        transfer_epoch_private=transfer_epoch_private,
        server_training_epoch=server_training_epoch,
        device=device,
        custom_action=custom_action,
    )

    return fedmd_api


def setup_fedgems(
    model_class,
    public_dataloader,
    local_dataloaders,
    test_dataloader,
    num_classes=20,
    channel=1,
    client_num=2,
    lr=0.01,
    epsilon=0.75,
    num_communication=10,
    epoch_client_on_localdataset=10,
    epoch_client_on_publicdataset=10,
    epoch_server_on_publicdataset=10,
    device="cpu",
    criterion=nn.CrossEntropyLoss(),
    weight_decay=0.01,
    input_dim=112 * 94,
    output_dim=20,
    round_decimal=None,
    custom_action=lambda x: x,
):
    # set up clients
    clients = [
        FedGEMSClient(
            model_class(input_dim=input_dim, output_dim=output_dim, channel=channel).to(
                device
            ),
            round_decimal=round_decimal,
            user_id=i,
            lr=lr,
            epsilon=epsilon,
        ).to(device)
        for i in range(client_num)
    ]
    client_optimizers = [
        optim.Adam(client.parameters(), lr=lr, weight_decay=weight_decay)
        for client in clients
    ]

    # set up the server
    global_model = model_class(
        input_dim=input_dim, output_dim=output_dim, channel=channel
    )
    global_model.to(device)
    server = FedGEMSServer(
        clients,
        global_model,
        len_public_dataloader=len(public_dataloader.dataset),
        self_evaluation_func=lambda y_pred, y: (
            torch.where(torch.argmax(y_pred, dim=1) == y)[0],
            torch.where(torch.argmax(y_pred, dim=1) != y)[0],
        ),
        output_dim=num_classes,
        lr=lr,
        epsilon=epsilon,
        device=device,
    ).to(device)
    server_optimizer = optim.Adam(server.parameters(), lr=lr, weight_decay=weight_decay)

    # set up FedGEMS
    fedgems_api = FedGEMSAPI(
        server=server,
        clients=clients,
        public_dataloader=public_dataloader,
        local_dataloaders=local_dataloaders,
        validation_dataloader=test_dataloader,
        server_optimizer=server_optimizer,
        client_optimizers=client_optimizers,
        criterion=criterion,
        num_communication=num_communication,
        epoch_client_on_localdataset=epoch_client_on_localdataset,
        epoch_client_on_publicdataset=epoch_client_on_publicdataset,
        epoch_server_on_publicdataset=epoch_server_on_publicdataset,
        device=device,
        custom_action=custom_action,
    )

    return fedgems_api


def setup_dsfl(
    model_class,
    public_dataloader,
    local_dataloaders,
    test_dataloader,
    channel=1,
    client_num=2,
    lr=0.01,
    num_communication=10,
    aggregation="ERA",
    era_temperature=0.1,
    epoch_local_training=2,
    epoch_local_distillation=2,
    epoch_global_distillation=1,
    device="cpu",
    criterion=nn.CrossEntropyLoss(),
    weight_decay=0.01,
    input_dim=112 * 94,
    output_dim=20,
    round_decimal=None,
    custom_action=lambda x: x,
):
    local_identities = [np.unique(dl.dataset.y).tolist() for dl in local_dataloaders]
    label2newlabel = {
        la: i for i, la in enumerate(np.unique(sum(local_identities, [])))
    }
    for local_dataloader in local_dataloaders:
        for i in range(local_dataloader.dataset.y.shape[0]):
            local_dataloader.dataset.y[i] = label2newlabel[
                local_dataloader.dataset.y[i]
            ]

    for i in range(public_dataloader.dataset.y.shape[0]):
        if public_dataloader.dataset.y[i] in label2newlabel:
            public_dataloader.dataset.y[i] = label2newlabel[
                public_dataloader.dataset.y[i]
            ]
        else:
            public_dataloader.dataset.y[i] = -1

    # set up clients
    clients = [
        DSFLClient(
            model_class(input_dim=input_dim, output_dim=output_dim, channel=channel).to(
                device
            ),
            public_dataloader,
            output_dim=output_dim,
            round_decimal=round_decimal,
            user_id=i,
            device=device,
        ).to(device)
        for i in range(client_num)
    ]
    client_optimizers = [
        optim.Adam(client.parameters(), lr=lr, weight_decay=weight_decay)
        for client in clients
    ]

    # set up the server
    global_model = model_class(
        input_dim=input_dim, output_dim=output_dim, channel=channel
    ).to(device)
    server = DSFLServer(
        clients,
        global_model,
        public_dataloader,
        aggregation=aggregation,
        era_temperature=era_temperature,
        device=device,
    ).to(device)
    server_optimizer = optim.Adam(server.parameters(), lr=lr, weight_decay=weight_decay)

    # set up FedGEMS
    fedgems_api = DSFLAPI(
        server,
        clients,
        public_dataloader,
        local_dataloaders,
        criterion,
        num_communication,
        device,
        server_optimizer,
        client_optimizers,
        validation_dataloader=test_dataloader,
        epoch_local_training=epoch_local_training,
        epoch_local_distillation=epoch_local_distillation,
        epoch_global_distillation=epoch_global_distillation,
        custom_action=custom_action,
    )

    return fedgems_api


def get_fedkd_api(
    fedkd_type,
    model_class,
    public_train_dataloader,
    local_train_dataloaders,
    test_dataloader,
    num_classes,
    client_num,
    channel,
    lr,
    num_communication,
    input_dim,
    device,
    config_fedkd,
    custom_action=None,
    target_celeblities_num=100,
):
    if fedkd_type == "FedGEMS":
        api = setup_fedgems(
            model_class,
            public_train_dataloader,
            local_train_dataloaders,
            test_dataloader,
            num_classes=num_classes,
            client_num=client_num,
            channel=channel,
            lr=lr,
            round_decimal=None,
            device=device,
            num_communication=num_communication,
            input_dim=input_dim,
            output_dim=num_classes,
            custom_action=custom_action,
            **config_fedkd,
        )
    elif fedkd_type == "FedMD":
        api = setup_fedmd(
            model_class,
            public_train_dataloader,
            local_train_dataloaders,
            test_dataloader,
            client_num=client_num,
            channel=channel,
            lr=lr,
            round_decimal=None,
            device=device,
            num_communication=num_communication,
            input_dim=input_dim,
            output_dim=num_classes,
            custom_action=custom_action,
            **config_fedkd,
        )
    elif fedkd_type == "DSFL":
        api = setup_dsfl(
            model_class,
            public_train_dataloader,
            local_train_dataloaders,
            test_dataloader,
            client_num=client_num,
            channel=channel,
            lr=lr,
            round_decimal=None,
            device=device,
            num_communication=num_communication,
            input_dim=input_dim,
            output_dim=target_celeblities_num,
            custom_action=custom_action,
            **config_fedkd,
        )
    else:
        raise NotImplementedError(f"{fedkd_type} is not supported")

    return api
