import torch
import torch.nn.functional as F
import sys, os, time

sys.path.append(os.pardir)
from models.autoencoder import AutoEncoder, AutoEncoder2, AutoEncoder3
from show_autoencoder import show_autoencoder_transform_result
from utils import cross_entropy_for_onehot
from utils import get_timestamp, sharpen, entropy

# device = 'cpu'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("currently using device:",device)

def label_to_one_hot(target, num_classes=100):
    target = torch.unsqueeze(target, 1)
    one_hot_target = torch.zeros(target.size(0), num_classes, device=target.device)
    one_hot_target.scatter_(1, target, 1)
    return one_hot_target


def train_batch(model, optimizer, code_y, hyperparameter_dict):
    label_y = torch.argmax(code_y, dim=1)
    # OH_y = label_to_one_hot(label_y, num_classes=dim)

    _lambda = hyperparameter_dict["_lambda"]

    # print("-"*100)
    # print("code_y", code_y)
    # =================== forward =====================
    code_y_hat, code_D_y = model(code_y)

    label_y_hat = torch.argmax(code_y_hat, dim=1)
    label_D_y = torch.argmax(code_D_y, dim=1)
    # print("code_D_y", code_D_y)
    loss_e = entropy(code_D_y)
    loss_p = cross_entropy_for_onehot(code_y_hat, code_y)
    # loss_p = criterion(code_y_hat, label_y)
    loss_n = cross_entropy_for_onehot(code_D_y, code_y)
    loss = 10 * loss_p - _lambda * loss_e - loss_n
    # loss = 10 * loss_p - _lambda * loss_e
    # loss = 10 * loss_p - loss_n

    # =================== backward ====================
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    train_acc_p = torch.sum(label_y_hat == label_y) / float(len(label_y))
    train_acc_n = torch.sum(label_D_y != label_y) / float(len(label_y))
    train_loss = loss.item()

    # =================== log =========================
    # print(f"loss_p:{loss_p.item()}, loss_e:{loss_e.item()}, loss_n:{loss_n.item()}")
    # loss_dict = {"loss_p": loss_p.item(), "loss_e": loss_e.item(), "loss_n": loss_n.item()}
    # loss_dict = {"loss_p": loss_p.item(), "loss_e": loss_e.item(), "loss_n": 0}
    loss_dict = {"loss_p": loss_p.item(), "loss_e": 0, "loss_n": loss_n.item()}
    return train_loss, train_acc_p, train_acc_n, loss_dict


if __name__ == '__main__':
    # num_classes: the input dim of auto-encoder
    num_classes = 20
    model = AutoEncoder(input_dim=num_classes, encode_dim= (2 + num_classes*6)).to(device)
    if num_classes==100:
        model = AutoEncoder(input_dim=num_classes, encode_dim= (2 + num_classes*2)).to(device)
    # model = AutoEncoder(input_dim=dim, encode_dim=2 + i * 10).to(device)
    # model = AutoEncoder(input_dim=num_classes, encode_dim=num_classes * 6).to(device)
    # model = AutoEncoder(input_dim=num_classes, encode_dim=num_classes).to(device)
    # learning_rate = 5e-4
    learning_rate = 5e-4
    if num_classes==100:
        learning_rate = 1e-5
    batch_size = 128
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    hyperparameter_dict = dict()
    # TODO: 0.1, 0.5, 1.0, 1.5, 2.0
    # _lambda is the hyper-parameter of CoAE
    _lambda = 0.5
    hyperparameter_dict["_lambda"] = (-1) * _lambda
    # epochs = 50
    # T = 0.05
    # epochs = 50  # train 100 classes
    # T = 0.05
    epochs = 30  # train 20 classes
    T = 0.028
    # epochs = 75  # train 10 classes
    # T = 0.025
    # T = 0.024
    # epochs = 20  # train 5 classes
    # T = 0.025
    # epochs = 50  # train 2 classes
    # T = 0.025

    # criterion = torch.nn.TripletMarginLoss(margin=2.0)
    criterion = torch.nn.CrossEntropyLoss()
    train_sample_size = 30000
    rand_train_x = torch.rand(train_sample_size, num_classes)
    train_y = sharpen(F.softmax(rand_train_x, dim=1), T=T)
    # print('train_y:', train_y)
    # print('train_y:', F.softmax(rand_train_x, dim=1))
    # time.sleep(100)
    # train_y = F.softmax(rand_train_x, dim=1)

    test_sample_size = 10000
    rand_test_x = torch.rand(test_sample_size, num_classes)
    test_y = sharpen(F.softmax(rand_test_x, dim=1), T=T)
    # test_y = F.softmax(rand_test_x, dim=1)

    print(f"train data: {train_y[0]}")
    for epoch in range(0, epochs + 1):
        iteration = 0
        train_loss = 0
        code_y = train_y.to(device)
        for batch_start_idx in range(0, code_y.shape[0], batch_size):
            iteration += 1
            # print(batch_start_idx, batch_start_idx + batch_size)
            batch_code_y = code_y[batch_start_idx:batch_start_idx + batch_size]
            train_loss, train_acc_p, train_acc_n, loss_dict = train_batch(model,
                                                                          optimizer,
                                                                          batch_code_y,
                                                                          hyperparameter_dict)

            if (iteration + 1) % 20 == 0:
                # validation on test data
                test_code_y = test_y.to(device)
                test_label_y = torch.argmax(test_code_y, dim=1)
                test_code_y_hat, test_code_D_y = model(test_code_y)
                # print("test_code_D_y: \n", test_code_D_y[0])
                tst_label_y_hat = torch.argmax(test_code_y_hat, dim=1)
                tst_label_D_y = torch.argmax(test_code_D_y, dim=1)

                test_acc_p = torch.sum(tst_label_y_hat == test_label_y) / float(len(test_label_y))
                test_acc_n = torch.sum(tst_label_D_y != test_label_y) / float(len(test_label_y))

                print(f"[INFO] epoch:{epoch}, iter:{iteration}, loss:{train_loss}")
                print(
                    f"[INFO] loss_p:{loss_dict['loss_p']}, loss_e:{loss_dict['loss_e']}, loss_n:{loss_dict['loss_n']}")
                print(f"[INFO]   train acc p : {train_acc_p}, train acc n : {train_acc_n}")
                print(f"[INFO]   test acc p : {test_acc_p}, test acc n : {test_acc_n}")

    timestamp = get_timestamp()
    model_name = f"autoencoder_{num_classes}_{_lambda}_{timestamp}"
    model_full_path = f"../trained_models/negative/{model_name}"
    model.save_model(model_full_path)
    print(f"[INFO] save model to:{model_full_path}")

    show_autoencoder_transform_result(model, num_classes)
