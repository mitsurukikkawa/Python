# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 11:24:50 2017

@author: izumiy
"""

import pywt
import numpy as np
import matplotlib.pyplot as plt

def create_scalogram_1(time_series, scales, wavelet, predict_time_inc, save_flag, ch_flag, height, width):
    """
    連続ウェーブレット変換を実行する関数
    終値を使用
    time_series      : 為替データ, 終値
    scales           : 使用するスケールをnumpy配列で指定する, スケールは分析に使用するウェーブレットの周波数に相当する, scalesが大きいと低周波, 小さいと高周波になる
    wavelet          : ウェーブレットの名前, 以下のいづれかを使用する
     'gaus1', 'gaus2', 'gaus3', 'gaus4', 'gaus5', 'gaus6', 'gaus7', 'gaus8', 'mexh', 'morl'
    predict_time_inc : 値動きを予測する時刻の増分
    save_flag        : save_flag=1 : CWT係数をcsvファイルとして保存する, save_flag=0 : CWT係数をcsvファイルとして保存しない
    ch_flag          : 使用するチャンネル数, ch_flag=1 : close
    height           : 画像の高さ num of time lines
    width            : 画像の幅  num of freq lines
    """

    """為替時系列データの読込み"""
    num_series_data = time_series.shape[0] # データ数の取得 
    print("number of the series data : " + str(num_series_data))
    close = time_series

    """連続ウェーブレット変換の実行"""
    # https://pywavelets.readthedocs.io/en/latest/ref/cwt.html
    print("carry out cwt...")
    time_start = 0
    time_end = time_start + height
    scalogram = np.empty((0, ch_flag, height, width))

    # hammingWindow = np.hamming(height)    # ハミング窓
    # hanningWindow = np.hanning(height)    # ハニング窓
    # blackmanWindow = np.blackman(height)  # ブラックマン窓
    # bartlettWindow = np.bartlett(height)  # バートレット窓

    while(time_end <= num_series_data - predict_time_inc):
        # print("time start " + str(time_start)) for debag
        temp_close = close[time_start:time_end]

        # 窓関数有り
        # temp_close = temp_close * hammingWindow

        # ミラー, データの前後に反転したデータを追加する
        mirror_temp_close = temp_close[::-1]
        x = np.append(mirror_temp_close, temp_close)
        temp_close = np.append(x, mirror_temp_close)

        temp_cwt_close, freq_close = pywt.cwt(temp_close, scales, wavelet)        # 連続ウェーブレット変換の実行
        temp_cwt_close = temp_cwt_close.T                                         # 転置 CWT(freq, time) ⇒ CWT(time, freq)

        # ミラー, 中央のデータのみ抽出する
        temp_cwt_close = temp_cwt_close[height:2*height,:]

        temp_cwt_close = np.reshape(temp_cwt_close, (-1, ch_flag, height, width)) # num_data, ch, height(time), width(freq)
        # print("temp_cwt_close_shape " + str(temp_cwt_close.shape)) # for debag
        scalogram = np.append(scalogram, temp_cwt_close, axis=0)
        # print("cwt_close_shape " + str(cwt_close.shape)) # for debag
        time_start = time_end
        time_end = time_start + height

    """ラベルの作成"""
    print("make label...")

    # 2つの配列を比較する方法
    last_time = num_series_data - predict_time_inc
    corrent_close = close[:last_time]
    predict_close = close[predict_time_inc:]
    label_array = predict_close > corrent_close
    # print(label_array[:30]) # for debag            

    """
    # whileを使う方法, 遅い
    label_array = np.array([])
    print(label_array)
    time_start = 0
    time_predict = time_start + predict_time_inc

    while(time_predict < num_series_data):
        if close[time_start] >= close[time_predict]:
            label = 0 # 下がる
        else:
            label = 1 # 上がる

        label_array = np.append(label_array, label)
        time_start = time_start + 1
        time_predict = time_start + predict_time_inc
    # print(label_array[:30]) # for debag
    """

    """label_array(time), timeがheightで割り切れる様にスライスする"""
    raw_num_shift = label_array.shape[0]
    num_shift = int(raw_num_shift / height) * height
    label_array = label_array[0:num_shift]

    """各スカログラムに対応したラベルの抽出, (データ数, ラベル)"""
    col = height - 1
    label_array = np.reshape(label_array, (-1, height))
    label_array = label_array[:, col]

    """ファイル出力"""
    if save_flag == 1:
        print("output the files")
        save_cwt_close = np.reshape(scalogram, (-1, width))
        np.savetxt("scalogram.csv", save_cwt_close, delimiter = ",")
        np.savetxt("label.csv", label_array.T, delimiter = ",")

    print("CWT is done")
    return scalogram, label_array, freq_close

def create_scalogram_5(time_series, scales, wavelet, predict_time_inc, save_flag, ch_flag, height, width):
    """
    連続ウェーブレット変換を実行する関数
    終値を使用
    time_series      : 為替データ, 終値
    scales           : 使用するスケールをnumpy配列で指定する, スケールは分析に使用するウェーブレットの周波数に相当する, scalesが大きいと低周波, 小さいと高周波になる
    wavelet          : ウェーブレットの名前, 以下のいづれかを使用する
     'gaus1', 'gaus2', 'gaus3', 'gaus4', 'gaus5', 'gaus6', 'gaus7', 'gaus8', 'mexh', 'morl'
    predict_time_inc : 値動きを予測する時刻の増分
    save_flag        : save_flag=1 : CWT係数をcsvファイルとして保存する, save_flag=0 : CWT係数をcsvファイルとして保存しない
    ch_flag          : 使用するチャンネル数, ch_flag=5 : start, high, low, close, volume
    height           : 画像の高さ num of time lines
    width            : 画像の幅  num of freq lines
    """

    """為替時系列データの読込み"""
    num_series_data = time_series.shape[0] # データ数の取得 
    print("number of the series data : " + str(num_series_data))
    start = time_series[:,0]
    high = time_series[:,1]
    low = time_series[:,2]
    close = time_series[:,3]
    volume = time_series[:,4]

    """連続ウェーブレット変換の実行"""
    # https://pywavelets.readthedocs.io/en/latest/ref/cwt.html
    print("carry out cwt...")
    time_start = 0
    time_end = time_start + height
    scalogram = np.empty((0, ch_flag, height, width))

    while(time_end <= num_series_data - predict_time_inc):
        # print("time start " + str(time_start)) for debag
        temp_start = start[time_start:time_end]
        temp_high = high[time_start:time_end]
        temp_low = low[time_start:time_end]
        temp_close = close[time_start:time_end]
        temp_volume = volume[time_start:time_end]

        temp_cwt_start, freq_start = pywt.cwt(temp_start, scales, wavelet)        # 連続ウェーブレット変換の実行
        temp_cwt_high, freq_high = pywt.cwt(temp_high, scales, wavelet)
        temp_cwt_low, freq_low = pywt.cwt(temp_low, scales, wavelet)
        temp_cwt_close, freq_close = pywt.cwt(temp_close, scales, wavelet)
        temp_cwt_volume, freq_volume = pywt.cwt(temp_volume, scales, wavelet)

        temp_cwt_start = temp_cwt_start.T                                         # 転置 CWT(freq, time) ⇒ CWT(time, freq)
        temp_cwt_high = temp_cwt_high.T
        temp_cwt_low = temp_cwt_low.T
        temp_cwt_close = temp_cwt_close.T
        temp_cwt_volume = temp_cwt_volume.T

        temp_cwt_start = np.reshape(temp_cwt_start, (-1, 1, height, width)) # num_data, ch, height(time), width(freq)
        temp_cwt_high = np.reshape(temp_cwt_high, (-1, 1, height, width))
        temp_cwt_low = np.reshape(temp_cwt_low, (-1, 1, height, width))
        temp_cwt_close = np.reshape(temp_cwt_close, (-1, 1, height, width))
        temp_cwt_volume = np.reshape(temp_cwt_volume, (-1, 1, height, width))
        # print("temp_cwt_close_shape " + str(temp_cwt_close.shape)) # for debag

        temp_cwt_start = np.append(temp_cwt_start, temp_cwt_high, axis=1)
        temp_cwt_start = np.append(temp_cwt_start, temp_cwt_low, axis=1)
        temp_cwt_start = np.append(temp_cwt_start, temp_cwt_close, axis=1)
        temp_cwt_start = np.append(temp_cwt_start, temp_cwt_volume, axis=1)
        # print("temp_cwt_start_shape " + str(temp_cwt_start.shape)) for debag

        scalogram = np.append(scalogram, temp_cwt_start, axis=0)
        # print("cwt_close_shape " + str(cwt_close.shape)) # for debag
        time_start = time_end
        time_end = time_start + height

    """ラベルの作成"""
    print("make label...")

    # 2つの配列を比較する方法
    last_time = num_series_data - predict_time_inc
    corrent_close = close[:last_time]
    predict_close = close[predict_time_inc:]
    label_array = predict_close > corrent_close
    # print(label_array[:30]) # for debag            

    """
    # whileを使う方法, 遅い
    label_array = np.array([])
    print(label_array)
    time_start = 0
    time_predict = time_start + predict_time_inc

    while(time_predict < num_series_data):
        if close[time_start] >= close[time_predict]:
            label = 0 # 下がる
        else:
            label = 1 # 上がる

        label_array = np.append(label_array, label)
        time_start = time_start + 1
        time_predict = time_start + predict_time_inc
    # print(label_array[:30]) # for debag
    """

    """label_array(time), timeがheightで割り切れる様にスライスする"""
    raw_num_shift = label_array.shape[0]
    num_shift = int(raw_num_shift / height) * height
    label_array = label_array[0:num_shift]

    """各スカログラムに対応したラベルの抽出, (データ数, ラベル)"""
    col = height - 1
    label_array = np.reshape(label_array, (-1, height))
    label_array = label_array[:, col]

    """ファイル出力"""
    if save_flag == 1:
        print("output the files")
        save_cwt_close = np.reshape(scalogram, (-1, width))
        np.savetxt("scalogram.csv", save_cwt_close, delimiter = ",")
        np.savetxt("label.csv", label_array.T, delimiter = ",")

    print("CWT is done")
    return scalogram, label_array, freq_close

def CWT_1(time_series, scales, wavelet, predict_time_inc, save_flag):
    """
    連続ウェーブレット変換を実行する関数
    終値を使用
    time_series      : 為替データ, 終値
    scales           : 使用するスケールをnumpy配列で指定する, スケールは分析に使用するウェーブレットの周波数に相当する, scalesが大きいと低周波, 小さいと高周波になる
    wavelet          : ウェーブレットの名前, 以下のいづれかを使用する
     'gaus1', 'gaus2', 'gaus3', 'gaus4', 'gaus5', 'gaus6', 'gaus7', 'gaus8', 'mexh', 'morl'
    predict_time_inc : 値動きを予測する時刻の増分
    save_flag        : save_flag=1 : CWT係数をcsvファイルとして保存する, save_flag=0 : CWT係数をcsvファイルとして保存しない
    """

    """為替時系列データの読込み"""
    num_series_data = time_series.shape[0] # データ数の取得 
    print("number of the series data : " + str(num_series_data))
    close = time_series

    """連続ウェーブレット変換の実行"""
    # https://pywavelets.readthedocs.io/en/latest/ref/cwt.html
    print("carry out cwt...")
    cwt_close, freq_close = pywt.cwt(close, scales, wavelet)

    # 転置 CWT(freq, time) ⇒ CWT(time, freq)
    cwt_close = cwt_close.T

    """ラベルの作成"""
    print("make label...")

    # 2つの配列を比較する方法
    last_time = num_series_data - predict_time_inc
    corrent_close = close[:last_time]
    predict_close = close[predict_time_inc:]
    label_array = predict_close > corrent_close
    # print(label_array[:30]) # for debag

    """
    # whileを使う方法
    label_array = np.array([])
    print(label_array)
    time_start = 0
    time_predict = time_start + predict_time_inc

    while(time_predict < num_series_data):
        if close[time_start] >= close[time_predict]:
            label = 0 # 下がる
        else:
            label = 1 # 上がる

        label_array = np.append(label_array, label)
        time_start = time_start + 1
        time_predict = time_start + predict_time_inc
    # print(label_array[:30]) # for debag
    """

    """ファイル出力"""
    if save_flag == 1:
        print("output the files")
        np.savetxt("CWT_close.csv", cwt_close, delimiter = ",")
        np.savetxt("label.csv", label_array.T, delimiter = ",")

    print("CWT is done")
    return [cwt_close], label_array, freq_close

def merge_CWT_1(cwt_list, label_array, height, width):
    """
    終値を使用
    cwt_list    : CWTの結果リスト
    label_array : ラベルを格納したnumpy配列
    height      : 画像の高さ num of time lines
    width       : 画像の幅  num of freq lines
    """
    print("merge CWT")

    cwt_close = cwt_list[0]  # 終値 CWT(time, freq)

    """CWT(time, freq), timeがheightで割り切れる様にスライスする"""
    raw_num_shift = cwt_close.shape[0]
    num_shift = int(raw_num_shift / height) * height
    cwt_close = cwt_close[0:num_shift]
    label_array = label_array[0:num_shift]

    """形状変更, (データ数, チャンネル, 高さ(time), 幅(freq))"""
    cwt_close = np.reshape(cwt_close, (-1, 1, height, width))

    """各スカログラムに対応したラベルの抽出, (データ数, ラベル)"""
    col = height - 1
    label_array = np.reshape(label_array, (-1, height))
    label_array = label_array[:, col]

    return cwt_close, label_array

def CWT_2(time_series, scales, wavelet, predict_time_inc, save_flag):
    """
    連続ウェーブレット変換を実行する関数
    終値,Volumeを使用
    time_series      : 為替データ, 終値, volume
    scales           : 使用するスケールをnumpy配列で指定する, スケールは分析に使用するウェーブレットの周波数に相当する, scalesが大きいと低周波, 小さいと高周波になる
    wavelet          : ウェーブレットの名前, 以下のいづれかを使用する
     'gaus1', 'gaus2', 'gaus3', 'gaus4', 'gaus5', 'gaus6', 'gaus7', 'gaus8', 'mexh', 'morl'
    predict_time_inc : 値動きを予測する時刻の増分
    save_flag        : save_flag=1 : CWT係数をcsvファイルとして保存する, save_flag=0 : CWT係数をcsvファイルとして保存しない
    """

    """為替時系列データの読込み"""
    num_series_data = time_series.shape[0] # データ数の取得 
    print("number of the series data : " + str(num_series_data))
    close = time_series[:,0]
    volume = time_series[:,1]

    """連続ウェーブレット変換の実行"""
    # https://pywavelets.readthedocs.io/en/latest/ref/cwt.html
    print("carry out cwt...")
    cwt_close, freq_close = pywt.cwt(close, scales, wavelet)
    cwt_volume, freq_volume = pywt.cwt(volume, scales, wavelet)

    # 転置 CWT(freq, time) ⇒ CWT(time, freq)
    cwt_close = cwt_close.T
    cwt_volume = cwt_volume.T

    """ラベルの作成"""
    print("make label...")

    # 2つの配列を比較する方法    
    last_time = num_series_data - predict_time_inc
    corrent_close = close[:last_time]
    predict_close = close[predict_time_inc:]
    label_array = predict_close > corrent_close
    # print(label_array[:30]) # for debag

    """
    # whileを使う方法
    label_array = np.array([])
    print(label_array)
    time_start = 0
    time_predict = time_start + predict_time_inc

    while(time_predict < num_series_data):
        if close[time_start] >= close[time_predict]:
            label = 0 # 下がる
        else:
            label = 1 # 上がる

        label_array = np.append(label_array, label)
        time_start = time_start + 1
        time_predict = time_start + predict_time_inc
    # print(label_array[:30]) # for debag
    """

    """ファイル出力"""
    if save_flag == 1:
        print("output the files")
        np.savetxt("CWT_close.csv", cwt_close, delimiter = ",")
        np.savetxt("CWT_volume.csv", cwt_volume, delimiter = ",")
        np.savetxt("label.csv", label_array.T, delimiter = ",")

    print("CWT is done")
    return [cwt_close, cwt_volume], label_array, freq_close

def merge_CWT_2(cwt_list, label_array, height, width):
    """
    終値,Volumeを使用
    cwt_list    : CWTの結果リスト
    label_array : ラベルを格納したnumpy配列
    height      : 画像の高さ num of time lines
    width       : 画像の幅  num of freq lines
    """
    print("merge CWT")

    cwt_close = cwt_list[0]  # 終値 CWT(time, freq)
    cwt_volume = cwt_list[1] # 出来高

    """CWT(time, freq), timeがheightで割り切れる様にスライスする"""
    raw_num_shift = cwt_close.shape[0]
    num_shift = int(raw_num_shift / height) * height
    cwt_close = cwt_close[0:num_shift]
    cwt_volume = cwt_volume[0:num_shift]
    label_array = label_array[0:num_shift]

    """形状変更, (データ数, チャンネル, 高さ(time), 幅(freq))"""
    cwt_close = np.reshape(cwt_close, (-1, 1, height, width))
    cwt_volume = np.reshape(cwt_volume, (-1, 1, height, width))

    """マージする"""
    cwt_close = np.append(cwt_close, cwt_volume, axis=1)

    """各スカログラムに対応したラベルの抽出, (データ数, ラベル)"""
    col = height - 1
    label_array = np.reshape(label_array, (-1, height))
    label_array = label_array[:, col]

    return cwt_close, label_array

def CWT_5(time_series, scales, wavelet, predict_time_inc, save_flag):
    """
    連続ウェーブレット変換を実行する関数
    始値，高値，安値，終値,Volumeを使用
    time_series      : 為替データ, 始値, 高値, 安値, 終値, volume
    scales           : 使用するスケールをnumpy配列で指定する, スケールは分析に使用するウェーブレットの周波数に相当する, scalesが大きいと低周波, 小さいと高周波になる
    wavelet          : ウェーブレットの名前, 以下のいづれかを使用する
     'gaus1', 'gaus2', 'gaus3', 'gaus4', 'gaus5', 'gaus6', 'gaus7', 'gaus8', 'mexh', 'morl'
    predict_time_inc : 値動きを予測する時刻の増分
    save_flag        : save_flag=1 : CWT係数をcsvファイルとして保存する, save_flag=0 : CWT係数をcsvファイルとして保存しない
    """

    """為替時系列データの読込み"""
    num_series_data = time_series.shape[0] # データ数の取得 
    print("number of the series data : " + str(num_series_data))
    start = time_series[:,0]
    high = time_series[:,1]
    low = time_series[:,2]
    close = time_series[:,3]
    volume = time_series[:,4]

    """連続ウェーブレット変換の実行"""
    # https://pywavelets.readthedocs.io/en/latest/ref/cwt.html
    print("carry out cwt...")
    cwt_start, freq_start = pywt.cwt(start, scales, wavelet)
    cwt_high, freq_high = pywt.cwt(high, scales, wavelet)
    cwt_low, freq_low = pywt.cwt(low, scales, wavelet)
    cwt_close, freq_close = pywt.cwt(close, scales, wavelet)
    cwt_volume, freq_volume = pywt.cwt(volume, scales, wavelet)

    # 転置 CWT(freq, time) ⇒ CWT(time, freq)
    cwt_start = cwt_start.T
    cwt_high = cwt_high.T
    cwt_low = cwt_low.T
    cwt_close = cwt_close.T
    cwt_volume = cwt_volume.T

    """ラベルの作成"""
    print("make label...")

    # 2つの配列を比較する方法
    last_time = num_series_data - predict_time_inc
    corrent_close = close[:last_time]
    predict_close = close[predict_time_inc:]
    label_array = predict_close > corrent_close
    # print(label_array.dtype) >>> bool

    """
    # whileを使う方法
    label_array = np.array([])
    print(label_array)
    time_start = 0
    time_predict = time_start + predict_time_inc

    while(time_predict < num_series_data):
        if close[time_start] >= close[time_predict]:
            label = 0 # 下がる
        else:
            label = 1 # 上がる

        label_array = np.append(label_array, label)
        time_start = time_start + 1
        time_predict = time_start + predict_time_inc
    # print(label_array[:30]) # for debag
    """

    """ファイル出力"""
    if save_flag == 1:
        print("output the files")
        np.savetxt("CWT_start.csv", cwt_start, delimiter = ",")
        np.savetxt("CWT_high.csv", cwt_high, delimiter = ",")
        np.savetxt("CWT_low.csv", cwt_low, delimiter = ",")
        np.savetxt("CWT_close.csv", cwt_close, delimiter = ",")
        np.savetxt("CWT_volume.csv", cwt_volume, delimiter = ",")
        np.savetxt("label.csv", label_array.T, delimiter = ",")

    print("CWT is done")
    return [cwt_start, cwt_high, cwt_low, cwt_close, cwt_volume], label_array, freq_close

def merge_CWT_5(cwt_list, label_array, height, width):
    """
    cwt_list    : CWTの結果リスト
    label_array : ラベルを格納したnumpy配列
    height      : 画像の高さ num of time lines
    width       : 画像の幅  num of freq lines
    """
    print("merge CWT")

    cwt_start = cwt_list[0]  # 始値
    cwt_high = cwt_list[1]   # 高値
    cwt_low = cwt_list[2]    # 安値
    cwt_close = cwt_list[3]  # 終値 CWT(time, freq)
    cwt_volume = cwt_list[4] # 出来高

    """CWT(time, freq), timeがheightで割り切れる様にスライスする"""
    raw_num_shift = cwt_close.shape[0]
    num_shift = int(raw_num_shift / height) * height
    cwt_start = cwt_start[0:num_shift]
    cwt_high = cwt_high[0:num_shift]
    cwt_low = cwt_low[0:num_shift]
    cwt_close = cwt_close[0:num_shift]
    cwt_volume = cwt_volume[0:num_shift]
    label_array = label_array[0:num_shift]

    """形状変更, (データ数, チャンネル, 高さ(time), 幅(freq))"""
    cwt_start = np.reshape(cwt_start, (-1, 1, height, width))
    cwt_high = np.reshape(cwt_high, (-1, 1, height, width))
    cwt_low = np.reshape(cwt_low, (-1, 1, height, width))
    cwt_close = np.reshape(cwt_close, (-1, 1, height, width))
    cwt_volume = np.reshape(cwt_volume, (-1, 1, height, width))

    """マージする"""
    cwt_start = np.append(cwt_start, cwt_high, axis=1)
    cwt_start = np.append(cwt_start, cwt_low, axis=1)
    cwt_start = np.append(cwt_start, cwt_close, axis=1)
    cwt_start = np.append(cwt_start, cwt_volume, axis=1)

    """各スカログラムに対応したラベルの抽出, (データ数, ラベル)"""
    col = height - 1
    label_array = np.reshape(label_array, (-1, height))
    label_array = label_array[:, col]
    # print(label_array.dtype) >>> bool

    return cwt_start, label_array

def make_scalogram(input_file_name, scales, wavelet, height, width, predict_time_inc, ch_flag, save_flag, over_lap_inc):
    """
    input_file_name  : 為替データのファイル名
    scales           : 使用するスケールをnumpy配列で指定する, スケールは分析に使用するウェーブレットの周波数に相当する, scalesが大きいと低周波, 小さいと高周波になる
    wavelet          : ウェーブレットの名前, 以下のいづれかを使用する
     'gaus1', 'gaus2', 'gaus3', 'gaus4', 'gaus5', 'gaus6', 'gaus7', 'gaus8', 'mexh', 'morl'
    predict_time_inc : 値動きを予測する時刻の増分
    height           : 画像の高さ num of time lines
    width            : 画像の幅  num of freq lines
    ch_flag          : 使用するチャンネル数, ch_flag=1:close, ch_flag=2:close and volume, ch_flag=5:start, high, low, close, volume
    save_flag        : save_flag=1 : CWT係数をcsvファイルとして保存する, save_flag=0 : CWT係数をcsvファイルとして保存しない
    over_lap_inc     : CWT開始時刻の増分
    """

    scalogram = np.empty((0, ch_flag, height, width)) # 全てのスカログラムとラベルを格納する配列
    label = np.array([])
    over_lap_start = 0
    over_lap_end = int((height - 1) / over_lap_inc) * over_lap_inc + 1

    if ch_flag==1:

        print("reading the input file...")    
        time_series = np.loadtxt(input_file_name, delimiter = ",", usecols = (5,), skiprows = 1) # 終値をnumpy配列として取得する

        for i in range(over_lap_start, over_lap_end, over_lap_inc):
            print("over_lap_start " + str(i))
            temp_time_series = time_series[i:] # CWTの開始時刻を変化させる
            cwt_list, label_array, freq = CWT_1(temp_time_series, scales, wavelet, predict_time_inc, save_flag) # CWTの実行
            temp_scalogram, temp_label = merge_CWT_1(cwt_list, label_array, height, width)                      # スカログラムの作成

            scalogram = np.append(scalogram, temp_scalogram, axis=0) # 全てのスカログラムとラベルを１つの配列にまとめる
            label = np.append(label, temp_label)

        print("scalogram_shape " + str(scalogram.shape))
        print("label shape " + str(label.shape))
        print("frequency " + str(freq))

    elif ch_flag==2:

        print("reading the input file...")    
        time_series = np.loadtxt(input_file_name, delimiter = ",", usecols = (5,6), skiprows = 1) # 終値,volumeをnumpy配列として取得する

        for i in range(over_lap_start, over_lap_end, over_lap_inc):
            print("over_lap_start " + str(i))
            temp_time_series = time_series[i:] # CWTの開始時刻を変化させる
            cwt_list, label_array, freq = CWT_2(temp_time_series, scales, wavelet, predict_time_inc, save_flag) # CWTの実行
            temp_scalogram, temp_label = merge_CWT_2(cwt_list, label_array, height, width)                      # スカログラムの作成

            scalogram = np.append(scalogram, temp_scalogram, axis=0) # 全てのスカログラムとラベルを１つの配列にまとめる
            label = np.append(label, temp_label)

        print("scalogram_shape " + str(scalogram.shape))
        print("label shape " + str(label.shape))
        print("frequency " + str(freq))

    elif ch_flag==5:

        print("reading the input file...")    
        time_series = np.loadtxt(input_file_name, delimiter = ",", usecols = (2,3,4,5,6), skiprows = 1) # 始値,高値,安値,終値,volumeをnumpy配列として取得する

        for i in range(over_lap_start, over_lap_end, over_lap_inc):
            print("over_lap_start " + str(i))
            temp_time_series = time_series[i:] # CWTの開始時刻を変化させる
            cwt_list, label_array, freq = CWT_5(temp_time_series, scales, wavelet, predict_time_inc, save_flag) # CWTの実行
            temp_scalogram, temp_label = merge_CWT_5(cwt_list, label_array, height, width)                      # スカログラムの作成

            scalogram = np.append(scalogram, temp_scalogram, axis=0) # 全てのスカログラムとラベルを１つの配列にまとめる
            label = np.append(label, temp_label)
            # print(temp_label.dtype) >>> bool
            # print(label.dtype)      >>> float64

        print("scalogram_shape " + str(scalogram.shape))
        print("label shape " + str(label.shape))
        print("frequency " + str(freq))

    label = label.astype(np.int)
    return scalogram, label

def merge_scalogram(input_file_name, scales, wavelet, height, width, predict_time_inc, ch_flag, save_flag, over_lap_inc):
    """
    input_file_name  : 為替データのファイル名
    scales           : 使用するスケールをnumpy配列で指定する, スケールは分析に使用するウェーブレットの周波数に相当する, scalesが大きいと低周波, 小さいと高周波になる
    wavelet          : ウェーブレットの名前, 以下のいづれかを使用する
     'gaus1', 'gaus2', 'gaus3', 'gaus4', 'gaus5', 'gaus6', 'gaus7', 'gaus8', 'mexh', 'morl'
    predict_time_inc : 値動きを予測する時刻の増分
    height           : 画像の高さ num of time lines
    width            : 画像の幅  num of freq lines
    ch_flag          : 使用するチャンネル数, ch_flag=1:close, ch_flag=2:close and volume, ch_flag=5:start, high, low, close, volume
    save_flag        : save_flag=1 : CWT係数をcsvファイルとして保存する, save_flag=0 : CWT係数をcsvファイルとして保存しない
    over_lap_inc     : CWT開始時刻の増分
    """

    scalogram = np.empty((0, ch_flag, height, width)) # 全てのスカログラムとラベルを格納する配列
    label = np.array([])
    over_lap_start = 0
    over_lap_end = int((height - 1) / over_lap_inc) * over_lap_inc + 1

    if ch_flag==1:

        print("reading the input file...")    
        time_series = np.loadtxt(input_file_name, delimiter = ",", usecols = (5,), skiprows = 1) # 終値をnumpy配列として取得する

        for i in range(over_lap_start, over_lap_end, over_lap_inc):
            print("over_lap_start " + str(i))
            temp_time_series = time_series[i:] # CWTの開始時刻を変化させる
            temp_scalogram, temp_label, freq = create_scalogram_1(temp_time_series, scales, wavelet, predict_time_inc, save_flag, ch_flag, height, width)
            scalogram = np.append(scalogram, temp_scalogram, axis=0) # 全てのスカログラムとラベルを１つの配列にまとめる
            label = np.append(label, temp_label)

        # print("scalogram_shape " + str(scalogram.shape))
        # print("label shape " + str(label.shape))
        # print("frequency " + str(freq))

    if ch_flag==5:

        print("reading the input file...")    
        time_series = np.loadtxt(input_file_name, delimiter = ",", usecols = (2,3,4,5,6), skiprows = 1) # 終値をnumpy配列として取得する

        for i in range(over_lap_start, over_lap_end, over_lap_inc):
            print("over_lap_start " + str(i))
            temp_time_series = time_series[i:] # CWTの開始時刻を変化させる
            temp_scalogram, temp_label, freq = create_scalogram_5(temp_time_series, scales, wavelet, predict_time_inc, save_flag, ch_flag, height, width)
            scalogram = np.append(scalogram, temp_scalogram, axis=0) # 全てのスカログラムとラベルを１つの配列にまとめる
            label = np.append(label, temp_label)

    label = label.astype(np.int)
    return scalogram, label, freq
