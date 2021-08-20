#模块调用
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import pywt

#小波去噪函数 固定式阈值

# 若入参判断 阈值处理方式threshold_usage = one' 则执行此函数 即过滤阈值与噪声标准差sigma无关
def wavelet_single_demensional_denoising_sqtwolog_one(data,noise_estimating_method,
                                         wavelet_type, decomposition_layer_number):
    wavelet_data = pywt.wavedec(data, wavelet=wavelet_type, level=int(decomposition_layer_number))
    for i in range(1, len(wavelet_data)):
        tmp = wavelet_data[i].copy()
        n = len(tmp)
        sigma = 1
        threshold_value = sigma * np.sqrt(2.0 * np.math.log(float(n), np.e))  # 求出最终阈值
        wavelet_data[i] = pywt.threshold(wavelet_data[i], threshold_value, mode=noise_estimating_method)
    data_wec = pywt.waverec(wavelet_data, wavelet_type)
    return data_wec

# 若入参判断 阈值处理方式threshold_usage = sln' 则执行此函数 即以第一层小波系数的标准差估计作为各层的标准差估计
def wavelet_single_demensional_denoising_sqtwolog_sln(data,noise_estimating_method,
                                         wavelet_type, decomposition_layer_number):
    wavelet_data = pywt.wavedec(data, wavelet=wavelet_type, level=int(decomposition_layer_number))
    cmp = wavelet_data[1].copy()
    n = len(cmp)
    cmp.sort() #排序
    medium = abs(cmp[int(n/2)]) #取中值以估计标准差
    sigma = medium / 0.6745
    for i in range(1, len(wavelet_data)):
        tmp = wavelet_data[i].copy()
        n = len(tmp)
        threshold_value = sigma * np.sqrt(2.0 * np.math.log(float(n), np.e))  # 求出最终阈值
        wavelet_data[i] = pywt.threshold(wavelet_data[i], threshold_value, mode=noise_estimating_method)
    data_wec = pywt.waverec(wavelet_data, wavelet_type)
    return data_wec

# 若入参判断 阈值处理方式threshold_usage = mln' 则执行此函数 即分别计算各层的噪声标准差
def wavelet_single_demensional_denoising_sqtwolog_mln(data,noise_estimating_method,
                                         wavelet_type, decomposition_layer_number):
    wavelet_data = pywt.wavedec(data, wavelet=wavelet_type, level=int(decomposition_layer_number))
    for i in range(1, len(wavelet_data)):
        tmp = wavelet_data[i].copy()
        n = len(tmp)
        tmp.sort()  # 排序
        medium = abs(tmp[int(n / 2)])  # 取中值以估计标准差
        sigma = medium / 0.6745
        threshold_value = sigma * np.sqrt(2.0 * np.math.log(float(n), np.e))  # 求出最终阈值
        wavelet_data[i] = pywt.threshold(wavelet_data[i], threshold_value, mode=noise_estimating_method)
    data_wec = pywt.waverec(wavelet_data, wavelet_type)
    return data_wec

wavelet_single_demensional_denoising_sqtwolog_function_dictionary = {'one':wavelet_single_demensional_denoising_sqtwolog_one,
                                                                'sln':wavelet_single_demensional_denoising_sqtwolog_sln,
                                                                'mln':wavelet_single_demensional_denoising_sqtwolog_mln}
# 固定式阈值去噪 主函数 从函数字典中提取相应参数对应的函数
def wavelet_single_demensional_denoising_sqtwolog(data,threshold_usage,noise_estimating_method,
                                         wavelet_type, decomposition_layer_number):
    return wavelet_single_demensional_denoising_sqtwolog_function_dictionary.get(threshold_usage)(data,
                                        noise_estimating_method, wavelet_type, decomposition_layer_number)









#主函数
path ='C:/Users/11038/Desktop/超声信号处理/totaldata_50mj_20210628_1.xls'#数据路径

#提取数据
data = pd.read_excel(path)
data = data.iloc[:, 1]
plt.plot(data)

print(data)

data_denoising_1 = wavelet_single_demensional_denoising_sqtwolog(data,threshold_usage='one',noise_estimating_method='soft',wavelet_type='db3',decomposition_layer_number=8)#调用小波去噪函数
data_denoising_2 = wavelet_single_demensional_denoising_sqtwolog(data,threshold_usage='sln',noise_estimating_method='soft',wavelet_type='db3',decomposition_layer_number=8)#调用小波去噪函数
data_denoising_3 = wavelet_single_demensional_denoising_sqtwolog(data,threshold_usage='mln',noise_estimating_method='soft',wavelet_type='db3',decomposition_layer_number=8)#调用小波去噪函数
plt.plot(data_denoising_3)#显示去噪结果
plt.plot(data_denoising_2)#显示去噪结果
plt.plot(data_denoising_1)#显示去噪结果
plt.show()
