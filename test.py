'''#path ='C:/Users/11038/Desktop/超声信号处理/totaldata_50mj_20210628_1.xls'#数据路径
path ='C:/Users/11038/Desktop/超声信号处理/totaldata_20210628_50mj.xls'#数据路径
file_path_len = len(path)
num = int(path[-8])
num += 1
path = path[:-8] + str(num) + path[-7:]
print(path)'''
import numpy as np

'''
import numpy as np
a = np.array([1,2,3,4,5])
b = max(a)
print(b)
'''
import pandas as pd
a = np.array([
             [1,1,1,1]])

b = np.array([
             [2,2,2,2]])
c = np.append(a,b,axis=0)
print(c)
df2 = pd.DataFrame(c.T
                    ,columns =['name','age',])
df2.to_excel(excel_writer='demo.xlsx', sheet_name='sheet_1')