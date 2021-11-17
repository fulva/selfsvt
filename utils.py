import torch as th
import numpy as np

th.set_printoptions(precision = 2, sci_mode = False)

def get_pairs(states, data_dict):
    label = []
    data1 = []
    data2 = []
    for i in range(len(states)):
        for j in range(len(states)):
            label.append(np.array(states[j]) - np.array(states[i]))
            data1.append(data_dict[states[i]])
            data2.append(data_dict[states[j]])
    label = th.FloatTensor(np.array(label)).float()
    data1 = th.FloatTensor(np.array(data1)/255).float()
    data2 = th.FloatTensor(np.array(data2)/255).float()
    return label, data1, data2

def get_single(states, data_dict, device):
    label = []
    data = []
    for i in range(len(states)):
        label.append(np.array(states[i]))
        data.append(data_dict[states[i]])

    label = th.FloatTensor(label).float()
    data = th.FloatTensor(np.array(data)/255).float()

    label = label.to(device)
    data = data.contiguous().view(-1, 32, 32, 3).permute(0,3,1,2).to(device)

    return label, data

def for_test(label, data1, data2, device):
    if len(data1) > 10000:
        idx = th.randperm(len(data1))[:10000]
    else:
        idx = th.randperm(len(data1))

    label = label[idx].to(device)
    data1 = data1[idx].contiguous().view(-1, 32, 32, 3).permute(0,3,1,2).to(device) 
    data2 = data2[idx].contiguous().view(-1, 32, 32, 3).permute(0,3,1,2).to(device)
    return label, data1, data2

def error(F, data1, label, method, data2 = None):
    if (len(data1)==0):
        error = "None"
    else:
        num = len(label[0])
        if data2 != None:
            pred = F(data2) - F(data1)
        else:
            pred = F(data1)
        if method == "mae":
            error = (th.abs(pred - label)).sum(1).mean()/num
            error = error.cpu().detach().numpy()
        elif method == "mde":
            error = (th.abs(th.round(pred) - label)).sum(1).mean()/num
            error = error.cpu().detach().numpy()
        else:
            print("    The method is not included, please check.")
    return error

def print_error(F, seen_fortest, unseen_fortest, train_fortest, test_fortest, method):
    seen_label, seen_data = seen_fortest
    unseen_label, unseen_data = unseen_fortest
    train_label, train_data1, train_data2 = train_fortest
    test_label, test_data1, test_data2 = test_fortest

    seen_error = error(F , seen_data,  seen_label, method)
    unseen_error = error(F, unseen_data, unseen_label, method)
    train_error = error(F, train_data1, train_label, method, train_data2)
    test_error = error(F, test_data1, test_label, method, test_data2)

    if train_error != "None":
        print("    train_"+method+": " +str(train_error), end='')
    if test_error != "None":
        print("    test_" +method +": " + str(test_error), end='')
    if seen_error != "None":
        print("    seen_" +method +": " + str(seen_error), end='')
    if unseen_error != "None":
        print("    unseen_" +method +": " + str(unseen_error))

    return train_error, test_error, seen_error, unseen_error

