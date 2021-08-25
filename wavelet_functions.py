#模块调用
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import pywt
import time
#小波去噪函数

# 固定式阈值sqtwolog
# 若入参判断 阈值处理方式threshold_usage = 'one' 则执行此函数 即过滤阈值与噪声标准差sigma无关
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

# 若入参判断 阈值处理方式threshold_usage = 'sln' 则执行此函数 即以第一层小波系数的标准差估计作为各层的标准差估计
def wavelet_single_demensional_denoising_sqtwolog_sln(data,noise_estimating_method,
                                         wavelet_type, decomposition_layer_number):
    wavelet_data = pywt.wavedec(data, wavelet=wavelet_type, level=int(decomposition_layer_number))
    cmp = wavelet_data[1].copy()
    n = len(cmp)
    for j in range(n):
        cmp[j] = abs(cmp[j])
    cmp.sort() #排序
    medium = cmp[int(n/2)] #取中值以估计标准差
    sigma = medium / 0.6745
    for i in range(1, len(wavelet_data)):
        tmp = wavelet_data[i].copy()
        n = len(tmp)
        threshold_value = sigma * np.sqrt(2.0 * np.math.log(float(n), np.e))  # 求出最终阈值
        wavelet_data[i] = pywt.threshold(wavelet_data[i], threshold_value, mode=noise_estimating_method)
    data_wec = pywt.waverec(wavelet_data, wavelet_type)
    return data_wec

# 若入参判断 阈值处理方式threshold_usage = 'mln' 则执行此函数 即分别计算各层的噪声标准差
def wavelet_single_demensional_denoising_sqtwolog_mln(data,noise_estimating_method,
                                         wavelet_type, decomposition_layer_number):
    wavelet_data = pywt.wavedec(data, wavelet=wavelet_type, level=int(decomposition_layer_number))
    for i in range(1, len(wavelet_data)):
        tmp = wavelet_data[i].copy()
        n = len(tmp)
        for j in range(n):
            tmp[j] = abs(tmp[j])
        tmp.sort()  # 排序
        medium = tmp[int(n / 2)]  # 取中值以估计标准差
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


#无偏风险估计阈值 rigrsure
# 若入参判断 阈值处理方式threshold_usage = 'one' 则执行此函数 即过滤阈值与噪声标准差sigma无关
def wavelet_single_demensional_denoising_rigrsure_one(data,noise_estimating_method,
                                         wavelet_type, decomposition_layer_number):
    wavelet_data = pywt.wavedec(data, wavelet=wavelet_type, level=int(decomposition_layer_number))
    sigma = 1
    for j in range(1, len(wavelet_data)):
        temprory_sum = 0   #用于储存f_k数组前n项和，以优化效率节约运行时间
        tmp = wavelet_data[j].copy()
        n = len(tmp)
        for i in range(n):
            tmp[i] = abs(tmp[i])
        tmp.sort()  # 排序
        f_k = []
        lamda_k = []
        risk_k = []
        for i in tmp:
            f_k.append(i**2)
            lamda_k.append(np.sqrt(i**2))
        for i in range(n):
            risk_k.append((n - 2 * i + temprory_sum + (n - i) * f_k[n - i - 1]) / n)  # 原公式risk_k[i] = (n- 2*i + sum(f_k[:i]) + (n-i)*f_k[n-i-1]) / n 但时间复杂度不可接受
            temprory_sum += f_k[i]
        min_risk = risk_k[0]
        min_risk_index = 0
        for i in range(n):
            if risk_k[i] < min_risk:
                min_risk = risk_k[i]
                min_risk_index = i
        threshold_value = sigma * lamda_k[min_risk_index]  # 求出最终阈值
        wavelet_data[j] = pywt.threshold(wavelet_data[j], threshold_value, mode=noise_estimating_method)
    data_wec = pywt.waverec(wavelet_data, wavelet_type)
    return data_wec

# 若入参判断 阈值处理方式threshold_usage = 'sln' 则执行此函数 即以第一层小波系数的标准差估计作为各层的标准差估计
def wavelet_single_demensional_denoising_rigrsure_sln(data,noise_estimating_method,
                                         wavelet_type, decomposition_layer_number):
    wavelet_data = pywt.wavedec(data, wavelet=wavelet_type, level=int(decomposition_layer_number))
    cmp = wavelet_data[1].copy()
    n = len(cmp)
    for i in range(n):
        cmp[i] = abs(cmp[i])
    cmp.sort()  # 排序
    medium = cmp[int(n / 2)]  # 取中值以估计标准差
    sigma = medium / 0.6745
    for j in range(1, len(wavelet_data)):
        temprory_sum = 0  #用于储存f_k数组前n项和，以优化效率节约运行时间
        tmp = wavelet_data[j].copy()
        n = len(tmp)
        for i in range(n):
            tmp[i] = abs(tmp[i])
        tmp.sort()  # 排序
        f_k = []
        lamda_k = []
        risk_k = []
        for i in tmp:
            f_k.append(i**2)
            lamda_k.append(np.sqrt(i**2))
        for i in range(n):
            risk_k.append((n - 2 * i + temprory_sum + (n - i) * f_k[n - i - 1]) / n)  # 原公式risk_k[i] = (n- 2*i + sum(f_k[:i]) + (n-i)*f_k[n-i-1]) / n 但时间复杂度不可接受
            temprory_sum += f_k[i]
        min_risk = risk_k[0]
        min_risk_index = 0
        for i in range(n):
            if risk_k[i] < min_risk:
                min_risk = risk_k[i]
                min_risk_index = i
        threshold_value = sigma * lamda_k[min_risk_index]  # 求出最终阈值
        wavelet_data[j] = pywt.threshold(wavelet_data[j], threshold_value, mode=noise_estimating_method)
    data_wec = pywt.waverec(wavelet_data, wavelet_type)
    return data_wec

# 若入参判断 阈值处理方式threshold_usage = 'mln' 则执行此函数 即分别计算各层的噪声标准差
def wavelet_single_demensional_denoising_rigrsure_mln(data,noise_estimating_method,
                                         wavelet_type, decomposition_layer_number):
    wavelet_data = pywt.wavedec(data, wavelet=wavelet_type, level=int(decomposition_layer_number))
    for j in range(1, len(wavelet_data)):
        temprory_sum = 0  #用于储存f_k数组前n项和，以优化效率节约运行时间
        tmp = wavelet_data[j].copy()
        n = len(tmp)
        for i in range(n):
            tmp[i] = abs(tmp[i])
        tmp.sort()  # 排序
        f_k = []
        lamda_k = []
        risk_k = []
        for i in range(n):
            f_k.append(tmp[i]**2)
            lamda_k.append(np.sqrt(f_k[i]))
        for i in range(n):
            risk_k.append((n- 2*i + temprory_sum + (n-i)*f_k[n-i-1]) / n) #原公式risk_k[i] = (n- 2*i + sum(f_k[:i]) + (n-i)*f_k[n-i-1]) / n 但时间复杂度不可接受
            temprory_sum += f_k[i]
        min_risk = risk_k[0]
        min_risk_index = 0
        for i in range(n):
            if risk_k[i] < min_risk:
                min_risk = risk_k[i]
                min_risk_index = i
        medium = tmp[int(n / 2)]  # 取中值以估计标准差
        sigma = medium / 0.6745
        threshold_value = sigma * lamda_k[min_risk_index]  # 求出最终阈值
        wavelet_data[j] = pywt.threshold(wavelet_data[j], threshold_value, mode=noise_estimating_method)
    data_wec = pywt.waverec(wavelet_data, wavelet_type)
    return data_wec

wavelet_single_demensional_denoising_rigrsure_function_dictionary = {'one':wavelet_single_demensional_denoising_rigrsure_one,
                                                                'sln':wavelet_single_demensional_denoising_rigrsure_sln,
                                                                'mln':wavelet_single_demensional_denoising_rigrsure_mln}

# 无偏风险估计阈值去噪 主函数 从函数字典中提取相应参数对应的函数
def wavelet_single_demensional_denoising_rigrsure(data,threshold_usage,noise_estimating_method,
                                         wavelet_type, decomposition_layer_number):
    return wavelet_single_demensional_denoising_rigrsure_function_dictionary.get(threshold_usage)(data,
                                        noise_estimating_method, wavelet_type, decomposition_layer_number)

#极大极小阈值 minimaxi
# 若入参判断 阈值处理方式threshold_usage = 'one' 则执行此函数 即过滤阈值与噪声标准差sigma无关
def wavelet_single_demensional_denoising_minimaxi_one(data,noise_estimating_method,
                                         wavelet_type, decomposition_layer_number):
    wavelet_data = pywt.wavedec(data, wavelet=wavelet_type, level=int(decomposition_layer_number))
    sigma = 1
    for i in range(1, len(wavelet_data)):
        tmp = wavelet_data[i].copy()
        n = len(tmp)
        if n <= 32:
            lamda = 0
        else:
            lamda = 0.3936 + 0.1829 * (np.math.log(n, np.e) / np.math.log(2, np.e))
        threshold_value = sigma * lamda  # 求出最终阈值
        wavelet_data[i] = pywt.threshold(wavelet_data[i], threshold_value, mode=noise_estimating_method)
    data_wec = pywt.waverec(wavelet_data, wavelet_type)
    return data_wec

# 若入参判断 阈值处理方式threshold_usage = 'sln' 则执行此函数 即以第一层小波系数的标准差估计作为各层的标准差估计
def wavelet_single_demensional_denoising_minimaxi_sln(data, noise_estimating_method,
                                                        wavelet_type, decomposition_layer_number):
    wavelet_data = pywt.wavedec(data, wavelet=wavelet_type, level=int(decomposition_layer_number))
    cmp = wavelet_data[1].copy()
    n = len(cmp)
    for j in range(n):
        cmp[j] = abs(cmp[j])
    cmp.sort()  # 排序
    medium = cmp[int(n / 2)]  # 取中值以估计标准差
    sigma = medium / 0.6745
    for i in range(1, len(wavelet_data)):
        tmp = wavelet_data[i].copy()
        n = len(tmp)
        if n <= 32:
            lamda = 0
        else:
            lamda = 0.3936 + 0.1829 * (np.math.log(n,np.e) / np.math.log(2,np.e))
        threshold_value = sigma * lamda  # 求出最终阈值
        wavelet_data[i] = pywt.threshold(wavelet_data[i], threshold_value, mode=noise_estimating_method)
    data_wec = pywt.waverec(wavelet_data, wavelet_type)
    return data_wec

# 若入参判断 阈值处理方式threshold_usage = 'mln' 则执行此函数 即分别计算各层的噪声标准差
def wavelet_single_demensional_denoising_minimaxi_mln(data, noise_estimating_method,
                                                        wavelet_type, decomposition_layer_number):
    wavelet_data = pywt.wavedec(data, wavelet=wavelet_type, level=int(decomposition_layer_number))
    for i in range(1, len(wavelet_data)):
        tmp = wavelet_data[i].copy()
        n = len(tmp)
        if n <= 32:
            lamda = 0
        else:
            lamda = 0.3936 + 0.1829 * (np.math.log(n,np.e) / np.math.log(2,np.e))
        for j in range(n):
            tmp[j] = abs(tmp[j])
        tmp.sort()  # 排序
        medium = tmp[int(n / 2)]  # 取中值以估计标准差
        sigma = medium / 0.6745
        threshold_value = sigma * lamda  # 求出最终阈值
        wavelet_data[i] = pywt.threshold(wavelet_data[i], threshold_value, mode=noise_estimating_method)
    data_wec = pywt.waverec(wavelet_data, wavelet_type)
    return data_wec

wavelet_single_demensional_denoising_minimaxi_function_dictionary = {'one':wavelet_single_demensional_denoising_minimaxi_one,
                                                                    'sln':wavelet_single_demensional_denoising_minimaxi_sln,
                                                                    'mln':wavelet_single_demensional_denoising_minimaxi_mln}

# 极大极小阈值去噪 主函数 从函数字典中提取相应参数对应的函数
def wavelet_single_demensional_denoising_minimaxi(data,threshold_usage,noise_estimating_method,
                                         wavelet_type, decomposition_layer_number):
    return wavelet_single_demensional_denoising_minimaxi_function_dictionary.get(threshold_usage)(data,
                                        noise_estimating_method, wavelet_type, decomposition_layer_number)

# 启发式阈值 heursure
# 若入参判断 阈值处理方式threshold_usage = 'one' 则执行此函数 即过滤阈值与噪声标准差sigma无关
def wavelet_single_demensional_denoising_heursure_one(data, noise_estimating_method,
                                                        wavelet_type, decomposition_layer_number):
    wavelet_data = pywt.wavedec(data, wavelet=wavelet_type, level=int(decomposition_layer_number))
    sigma = 1
    for i in range(1, len(wavelet_data)):
        tmp = wavelet_data[i].copy()
        n = len(tmp)
        crit = np.math.sqrt((np.math.log(n,np.e) / np.math.log(2,np.e))**3 / n)
        tmp = np.array(tmp)
        tmp_square = tmp ** 2
        eta = (np.sum(tmp_square) - n) / n
        if eta < crit: # 此时选用sqtwolog阈值
            threshold_value = sigma * np.sqrt(2.0 * np.math.log(float(n), np.e))  # 求出最终阈值
            wavelet_data[i] = pywt.threshold(wavelet_data[i], threshold_value, mode=noise_estimating_method)
        else:# 此时选用sqtwolog和rigrsure中较小的那个
            #sqtwolog
            threshold_value_sqtwolog = sigma * np.sqrt(2.0 * np.math.log(float(n), np.e))
            #rigrsure
            temprory_sum = 0
            f_k = []
            lamda_k = []
            risk_k = []
            for j in range(n):
                f_k.append(tmp[j] ** 2)
                lamda_k.append(np.sqrt(f_k[j]))
            for j in range(n):
                risk_k.append((n - 2 * j + temprory_sum + (n - j) * f_k[n - j - 1]) / n)  # 原公式risk_k[i] = (n- 2*i + sum(f_k[:i]) + (n-i)*f_k[n-i-1]) / n 但时间复杂度不可接受
                temprory_sum += f_k[j]
            min_risk = risk_k[0]
            min_risk_index = 0
            for j in range(n):
                if risk_k[j] < min_risk:
                    min_risk = risk_k[j]
                    min_risk_index = j
            threshold_value_rigrsure = sigma * lamda_k[min_risk_index]
            # 两个阈值中取小的那个
            threshold_value = min(threshold_value_sqtwolog,threshold_value_rigrsure)
            wavelet_data[i] = pywt.threshold(wavelet_data[i], threshold_value, mode=noise_estimating_method)
    data_wec = pywt.waverec(wavelet_data, wavelet_type)
    return data_wec

# 若入参判断 阈值处理方式threshold_usage = 'sln' 则执行此函数 即以第一层小波系数的标准差估计作为各层的标准差估计
def wavelet_single_demensional_denoising_heursure_sln(data, noise_estimating_method,
                                                        wavelet_type, decomposition_layer_number):
    wavelet_data = pywt.wavedec(data, wavelet=wavelet_type, level=int(decomposition_layer_number))
    cmp = wavelet_data[1].copy()
    n = len(cmp)
    for i in range(n):
        cmp[i] = abs(cmp[i])
    cmp.sort()  # 排序
    medium = cmp[int(n / 2)]  # 取中值以估计标准差
    sigma = medium / 0.6745
    for i in range(1, len(wavelet_data)):
        tmp = wavelet_data[i].copy()
        n = len(tmp)
        crit = np.math.sqrt((np.math.log(n,np.e) / np.math.log(2,np.e))**3 / n)
        tmp = np.array(tmp)
        tmp_square = tmp ** 2
        eta = (np.sum(tmp_square) - n) / n
        if eta < crit: # 此时选用sqtwolog阈值
            threshold_value = sigma * np.sqrt(2.0 * np.math.log(float(n), np.e))  # 求出最终阈值
            wavelet_data[i] = pywt.threshold(wavelet_data[i], threshold_value, mode=noise_estimating_method)
        else:# 此时选用sqtwolog和rigrsure中较小的那个
            #sqtwolog
            threshold_value_sqtwolog = sigma * np.sqrt(2.0 * np.math.log(float(n), np.e))
            #rigrsure
            temprory_sum = 0
            f_k = []
            lamda_k = []
            risk_k = []
            for j in range(n):
                f_k.append(tmp[j] ** 2)
                lamda_k.append(np.sqrt(f_k[j]))
            for j in range(n):
                risk_k.append((n - 2 * j + temprory_sum + (n - j) * f_k[n - j - 1]) / n)  # 原公式risk_k[i] = (n- 2*i + sum(f_k[:i]) + (n-i)*f_k[n-i-1]) / n 但时间复杂度不可接受
                temprory_sum += f_k[j]
            min_risk = risk_k[0]
            min_risk_index = 0
            for j in range(n):
                if risk_k[j] < min_risk:
                    min_risk = risk_k[j]
                    min_risk_index = j
            threshold_value_rigrsure = sigma * lamda_k[min_risk_index]
            # 两个阈值中取小的那个
            threshold_value = min(threshold_value_sqtwolog,threshold_value_rigrsure)
            wavelet_data[i] = pywt.threshold(wavelet_data[i], threshold_value, mode=noise_estimating_method)
    data_wec = pywt.waverec(wavelet_data, wavelet_type)
    return data_wec
aaaaa = []
# 若入参判断 阈值处理方式threshold_usage = 'mln' 则执行此函数 即分别计算各层的噪声标准差
def wavelet_single_demensional_denoising_heursure_mln(data, noise_estimating_method,
                                                        wavelet_type, decomposition_layer_number):
    wavelet_data = pywt.wavedec(data, wavelet=wavelet_type, level=int(decomposition_layer_number))
    for i in range(1, len(wavelet_data)):
        tmp = wavelet_data[i].copy()
        n = len(tmp)
        crit = np.math.sqrt((np.math.log(n,np.e) / np.math.log(2,np.e))**3 / n)
        tmp = np.array(tmp)
        tmp_square = tmp ** 2
        eta = (np.sum(tmp_square) - n) / n
        eta = abs(eta)
        if eta < crit: # 此时选用sqtwolog阈值
            aaaaa.append(1)
            for j in range(n):
                tmp[j] = abs(tmp[j])
            tmp.sort()  # 排序
            medium = tmp[int(n / 2)]  # 取中值以估计标准差
            sigma = medium / 0.6745
            threshold_value = sigma * np.sqrt(2.0 * np.math.log(float(n), np.e))  # 求出最终阈值
            wavelet_data[i] = pywt.threshold(wavelet_data[i], threshold_value, mode=noise_estimating_method)
        else:# 此时选用sqtwolog和rigrsure中较小的那个
            #sqtwolog
            for j in range(n):
                tmp[j] = abs(tmp[j])
            tmp.sort()  # 排序
            medium = tmp[int(n / 2)]  # 取中值以估计标准差
            sigma = medium / 0.6745
            threshold_value_sqtwolog = sigma * np.sqrt(2.0 * np.math.log(float(n), np.e))
            #rigrsure
            temprory_sum = 0
            f_k = []
            lamda_k = []
            risk_k = []
            for j in range(n):
                f_k.append(tmp[j] ** 2)
                lamda_k.append(np.sqrt(f_k[j]))
            for j in range(n):
                risk_k.append((n - 2 * j + temprory_sum + (n - j) * f_k[n - j - 1]) / n)  # 原公式risk_k[i] = (n- 2*i + sum(f_k[:i]) + (n-i)*f_k[n-i-1]) / n 但时间复杂度不可接受
                temprory_sum += f_k[j]
            min_risk = risk_k[0]
            min_risk_index = 0
            for j in range(n):
                if risk_k[j] < min_risk:
                    min_risk = risk_k[j]
                    min_risk_index = j
            medium = tmp[int(n / 2)]  # 取中值以估计标准差
            sigma = medium / 0.6745
            threshold_value_rigrsure = sigma * lamda_k[min_risk_index]
            # 两个阈值中取小的那个
            if threshold_value_sqtwolog < threshold_value_rigrsure:
                aaaaa.append(2)
            else:
                aaaaa.append(3)
            threshold_value = min(threshold_value_sqtwolog,threshold_value_rigrsure)
            wavelet_data[i] = pywt.threshold(wavelet_data[i], threshold_value, mode=noise_estimating_method)
    data_wec = pywt.waverec(wavelet_data, wavelet_type)
    return data_wec

wavelet_single_demensional_denoising_heursure_function_dictionary = {'one':wavelet_single_demensional_denoising_heursure_one,
                                                                     'sln':wavelet_single_demensional_denoising_heursure_sln,
                                                                     'mln':wavelet_single_demensional_denoising_heursure_mln}

# 启发式阈值去噪 主函数 从函数字典中提取相应参数对应的函数
def wavelet_single_demensional_denoising_heursure(data,threshold_usage,noise_estimating_method,
                                         wavelet_type, decomposition_layer_number):
    return wavelet_single_demensional_denoising_heursure_function_dictionary.get(threshold_usage)(data,
                                        noise_estimating_method, wavelet_type, decomposition_layer_number)
'''
#主函数
path ='C:/Users/11038/Desktop/超声信号处理/totaldata_50mj_20210628_1.xls'#数据路径

#提取数据
data = pd.read_excel(path)
data = data.iloc[:, 1]
data = np.array(data)
data = data[2500:12500]
#plt.plot(data)

print(data)


data_denoising_1 = wavelet_single_demensional_denoising_rigrsure(data,threshold_usage='one',noise_estimating_method='soft',wavelet_type='db3',decomposition_layer_number=8)#调用小波去噪函数

data_denoising_2 = wavelet_single_demensional_denoising_rigrsure(data,threshold_usage='sln',noise_estimating_method='soft',wavelet_type='db3',decomposition_layer_number=8)#调用小波去噪函数

data_denoising_3 = wavelet_single_demensional_denoising_rigrsure(data,threshold_usage='mln',noise_estimating_method='soft',wavelet_type='db3',decomposition_layer_number=8)#调用小波去噪函数

data_denoising_4 = wavelet_single_demensional_denoising_sqtwolog(data,threshold_usage='one',noise_estimating_method='soft',wavelet_type='db3',decomposition_layer_number=8)#调用小波去噪函数

data_denoising_5 = wavelet_single_demensional_denoising_sqtwolog(data,threshold_usage='sln',noise_estimating_method='soft',wavelet_type='db3',decomposition_layer_number=8)#调用小波去噪函数

data_denoising_6 = wavelet_single_demensional_denoising_sqtwolog(data,threshold_usage='mln',noise_estimating_method='soft',wavelet_type='db3',decomposition_layer_number=8)#调用小波去噪函数



data_denoising_7 = wavelet_single_demensional_denoising_minimaxi(data,threshold_usage='one',noise_estimating_method='soft',wavelet_type='db3',decomposition_layer_number=8)#调用小波去噪函数
data_denoising_8 = wavelet_single_demensional_denoising_minimaxi(data,threshold_usage='sln',noise_estimating_method='soft',wavelet_type='db3',decomposition_layer_number=8)#调用小波去噪函数
data_denoising_9 = wavelet_single_demensional_denoising_minimaxi(data,threshold_usage='mln',noise_estimating_method='soft',wavelet_type='db3',decomposition_layer_number=8)#调用小波去噪函数


data_denoising_10 = wavelet_single_demensional_denoising_heursure(data,threshold_usage='one',noise_estimating_method='soft',wavelet_type='db3',decomposition_layer_number=8)#调用小波去噪函数
data_denoising_11 = wavelet_single_demensional_denoising_heursure(data,threshold_usage='sln',noise_estimating_method='soft',wavelet_type='db3',decomposition_layer_number=8)#调用小波去噪函数
data_denoising_12 = wavelet_single_demensional_denoising_heursure(data,threshold_usage='mln',noise_estimating_method='soft',wavelet_type='db3',decomposition_layer_number=8)#调用小波去噪函数

#plt.plot(data_denoising_3)#显示去噪结果
#plt.plot(data_denoising_2)#显示去噪结果
#plt.plot(data_denoising_1)#显示去噪结果

#plt.plot(data_denoising_6)#显示去噪结果
#plt.plot(data_denoising_5)#显示去噪结果
#plt.plot(data_denoising_4)#显示去噪结果

#plt.plot(data_denoising_9)#显示去噪结果
#plt.plot(data_denoising_8)#显示去噪结果
#plt.plot(data_denoising_7)#显示去噪结果

plt.plot(data_denoising_12)#显示去噪结果
plt.plot(data_denoising_6)#显示去噪结果
#plt.plot(data_denoising_11)#显示去噪结果
#plt.plot(data_denoising_10)#显示去噪结果
print(aaaaa)
plt.show()
'''