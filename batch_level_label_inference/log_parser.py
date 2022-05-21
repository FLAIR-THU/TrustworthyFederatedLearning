import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


def read_log(log_dir, total_epoch):
    with open(log_dir, 'r') as f:
        lines = f.readlines()
    
    paramers = []
    paramer_count = -1
    train_acc = []
    test_acc = []
    last_parameter = -1.0
    continue_flag = False
    for line in lines:
        if 'num_exp' in line:
            temp = line.strip("\n").replace(" ", "").split(",")
            current_parameter = temp[3].split(":")[1]
            if current_parameter == last_parameter:
                continue_flag = True
            else:
                last_parameter = current_parameter
                continue_flag = False
            if not continue_flag:
                paramers.append(current_parameter)
                paramer_count += 1
                train_acc.append([0.0]*total_epoch)
                test_acc.append([0.0]*total_epoch)
        elif 'train_loss' in line:
            if not continue_flag:
                temp = line.strip("\n").split(" ")
                epoch_index = int(temp[1][:-1])
                assert( 0<= epoch_index < total_epoch )
                # print("temp =",temp)
                train_acc[paramer_count][epoch_index] = float(temp[-2].split(":")[1])
                test_acc[paramer_count][epoch_index] = float(temp[-1].split(":")[1])
    print(paramers)
    # print(train_acc[2])
    # print(test_acc[3])
    
    return paramers,train_acc,test_acc


def plot_train_validate_acc_line(log_dir, totol_epoch, ylim=[0.0,1.0]):
    paramers,train_acc,test_acc = read_log(log_dir, totol_epoch)
    count = len(paramers)
    # fig, ax = plt.subplots(nrows=count,ncols=1)
    # line_list = [None] * count
    # colors = ['red','blue','green','mediumslateblue','orange']
    # for ic in range(count):
    #     ax[ic].plot(train_acc[ic],color=colors[ic],linestyle='--',label=paramers[ic]+" train")
    #     ax[ic].plot(test_acc[ic],color=colors[ic],label=paramers[ic]+" test")
    #     ax[ic].legend()
    #     ax[ic].set_xlabel("epochs", fontsize=10)
    #     # ax[ic].set_ylabel("Accuracy, dash for train, solid for test", fontsize=10)
    # plt.savefig(log_dir[:-3]+"png",dpi=200)
    # plt.close()

    fig, ax = plt.subplots(nrows=1,ncols=1,sharey=True,sharex=True,figsize=(int(3*totol_epoch/30),5))
    # plt.figure()
    line_list = [None] * count
    colors = ['red','blue','green','mediumslateblue','orange','aquamarine','dodgerblue']
    for ic in range(count):
        ax.plot(train_acc[ic],color=colors[ic],linestyle='--',label=paramers[ic]+" train")
        ax.plot(test_acc[ic],color=colors[ic],label=paramers[ic]+" test")
    ax.legend(fontsize=7)
    ax.set_xlabel("epochs", fontsize=10)
    ax.set_ylabel("Accuracy, dash for train, solid for test", fontsize=10)
    ax.set_ylim(ylim)
    # ax.yaxis.set_major_locator(mtick.MultipleLocator(5))
    plt.savefig(log_dir[:-3]+"png",dpi=200)
    plt.close()



if __name__ == '__main__':
    datasets = ['nuswide'] # 'nuswide','cifar'
    defense_methods = ['laplace']
    epochs = [30] # ,100

    # for dataset in datasets:
    #     for defense in defense_methods:
    #         for epoch in epochs:
    #             plot_train_validate_acc_line('{}_{}_log_{}.txt'.format(dataset,defense,epoch),epoch)
    # plot_train_validate_acc_line('cifar100_log.txt',1000)
    # plot_train_validate_acc_line('nuswide_log.txt',100)
    # plot_train_validate_acc_line('cifar10_log.txt',300)
    # plot_train_validate_acc_line('nuswide_water_animal_log.txt',100,[0.7,1.0])
    # plot_train_validate_acc_line('nuswide_clouds_person_log.txt',100,[0.7,1.0])
    # plot_train_validate_acc_line('log_20classes_test.txt',100,[0.4,1.0])
    plot_train_validate_acc_line('cifar10_main_task_gaussian_log_repeat10.txt',100,[0.2,1.0])
    plot_train_validate_acc_line('cifar10_main_task_log_repeat10.txt',200,[0.2,1.0])
    # plot_train_validate_acc_line('log_20classes_test.txt',100,[0.4,1.0])
