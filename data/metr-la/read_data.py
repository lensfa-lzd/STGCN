import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


def calc_time_index(test):
    test['timeofday'] = None

    for i in range(test.shape[0]):
        time_ = test.loc[i][0]
        hour, min = time_.hour, time_.minute
        index = (60 * hour + min) / 5
        # print(hour, min)
        test.timeofday[i] = int(index)

    test['dayofweek'] = test["time_index"].dt.dayofweek
    test["week_name"] = test["time_index"].dt.day_name()
    print(test)

    h5_sd = pd.HDFStore('time_index_.h5', 'w')
    h5_sd['data'] = test
    h5_sd.close()


def calc_TimeEmbedding(time_index, num_TimeofDay, num_DayofWeek=7):
    data = pd.read_hdf('time_index.h5')
    dayofweek = time_index.dayofweek.values
    dayofweek = F.one_hot(torch.tensor(dayofweek), num_classes=num_DayofWeek)

    timeofday = time_index.timeofday.values.astype(np.int64)
    timeofday = F.one_hot(torch.tensor(timeofday), num_classes=num_TimeofDay)

    return torch.cat(dayofweek, timeofday).float()


data = torch.load('vel.pth')
print(data.shape)
