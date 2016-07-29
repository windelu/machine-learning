
#learning pandas
import pandas as pa
import numpy as np

test_path = "/home/xuqian/mxnet2/data/titinic_test.csv"
# data = pa.read_csv(test_path)     # pandas.core.frame.DataFrame
data = pa.read_csv(test_path).values  #<type 'numpy.ndarray'>
data2 = np.delete(data,[0,2,7,8,9,10],axis=1)
for i in data2:
    if i[1] =='male':
        i[1] = 0
    else:
        i[1] =1


print data2



