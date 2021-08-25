#path ='C:/Users/11038/Desktop/超声信号处理/totaldata_50mj_20210628_1.xls'#数据路径
path ='C:/Users/11038/Desktop/超声信号处理/totaldata_20210628_50mj.xls'#数据路径
file_path_len = len(path)
num = int(path[-8])
num += 1
path = path[:-8] + str(num) + path[-7:]
print(path)