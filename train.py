import numpy as np
import torch as th
import random

from utils import get_pairs, for_test, get_single, print_error

random.seed(10)
th.manual_seed(10)

def train_siamese(F, states, data_dict, device, scene, num, seen, steps, bs):

    seen_ratio = seen
    seen = int(np.round(seen_ratio*len(states)/100))

    train_label, train_data1, train_data2 = get_pairs(states[:seen], data_dict)
    train_fortest = for_test(train_label, train_data1, train_data2, device)

    test_label, test_data1, test_data2  = get_pairs(states[seen:], data_dict)
    test_fortest  = for_test(test_label, test_data1, test_data2, device)

    seen_fortest = get_single(states[:seen],data_dict, device)
    unseen_fortest = get_single(states[seen:], data_dict, device)


    optimizer = th.optim.Adam(F.parameters(), lr=0.0001)

    for step in range(steps):
        optimizer.zero_grad()
        if bs < len(train_label): 
            idx = th.randperm(len(train_label))[:bs]
        else:
            idx = th.randperm(bs)%len(train_label)
        #Train
        label = train_label[idx].to(device)
        data1 = train_data1[idx].contiguous().view(bs, 32, 32, 3).permute(0,3,1,2).to(device)
        data2 = train_data2[idx].contiguous().view(bs, 32, 32, 3).permute(0,3,1,2).to(device)

        train_pred = F(data2) - F(data1)

        loss = ((train_pred - label)**2).sum(1).mean()/num

        loss.backward()

        optimizer.step()
        #Test
        if (step+1)%1000 == 0: 
            print("\n    ------------------"+str(step)+": " +str((loss).cpu().detach().numpy()) + "------------------------------")
            mae = print_error(F, seen_fortest, unseen_fortest, train_fortest, test_fortest,"mae")
            mde = print_error(F, seen_fortest, unseen_fortest, train_fortest, test_fortest,"mde")

