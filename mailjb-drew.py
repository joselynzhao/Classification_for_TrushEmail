# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 20:29:51 2017

@author: Lony
"""
import os
import jieba
import re
import numpy as np

import sys
import copy

import datetime
import pytesseract
from PIL import Image

import matplotlib.pyplot as plt
import matplotlib as mpl

import imageio

windowsize = 5
trashdic = {}
normaldic = {}
fnormalnum = float(0)
ftrashnum = float(0)

zh_pattern = re.compile(u'[\u4e00-\u9fa5]+')
def contain_zh(uword):
    '''
    判断传入字符串是否包含中文
    :param word: 待判断字符串
    :return: True:包含中文  False:不包含中文
    '''
    global zh_pattern
    match = zh_pattern.search(uword)
    return match

def create_gif(gif_name, path, duration = 0.1):
    '''
    生成gif文件，原始图片仅支持png格式
    gif_name ： 字符串，所生成的 gif 文件名，带 .gif 后缀
    path :      需要合成为 gif 的图片所在路径
    duration :  gif 图像时间间隔
    '''

    frames = []
    pngFiles = os.listdir(path)
    image_list = [os.path.join(path, f) for f in pngFiles]
    for image_name in image_list:
        # 读取 png 图像文件
        frames.append(imageio.imread(image_name))
    # 保存为 gif 
    imageio.mimsave(gif_name, frames, 'GIF', duration = duration)
    return

def draw(x, plot_x,plot_y,path):
    plt.title('Accuracy')
    plt.plot(x,plot_x,linewidth=1,c='yellow',label='normal')
    plt.plot(x,plot_y,linewidth=1,c='red',label='trash')
    #设置，x,y坐标标签和它的大小
    plt.xlabel('mails', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    #设置刻度数字的大小
    plt.tick_params(axis='both', labelsize=10)
    #plt.show()
    print(path+'.png')
    plt.savefig(path+'.png')
    plt.close()

def stopwordlist(filepath):
    lis = []
    doc = open(filepath,'r',errors='ignore')
    while True:
        sent = doc.readline()
        sent = sent.strip()
        if sent == '':
            break
        lis.append(sent)
    doc.close()
    return lis

def writelist(lis=[],filedir=''):
  fout = open(filedir,'w')
  for i in range(0,len(lis)):
    fout.write('%.6f\n'%(lis[i]))
  fout.close()

def getTrainset(norpath,trapath):
    ##获取已经训练过的训练集，包括正常邮件和垃圾邮件中出现词及词频
    nordic = {}
    tradic = {}
    doc = open(norpath,'r')
    while True:
        sent = doc.readline()
        if sent == '':
            break
        lis = sent.split()
        #print(lis[0])
        nordic[lis[0]] = float(lis[1])
    doc.close()
    doc = open(trapath,'r')
    while True:
        sent = doc.readline()
        if sent == '':
            break
        lis = sent.split()
        tradic[lis[0]] = float(lis[1])
    doc.close()
    return nordic,tradic

def GetTrainedDic(wordset, raw_nor_dic, raw_tra_dic, trainedSetLen):
    maxpahdic = {}
    for w in wordset:
        if raw_nor_dic.get(w):
            maxpahdic[w] = raw_nor_dic[w]
        else:
            maxpahdic[w] = 0
        if raw_tra_dic.get(w) and (maxpahdic[w] < raw_tra_dic[w]):
                maxpahdic[w] = raw_tra_dic[w]
    
    nodic = {}
    trdic = {}
    fnormalnum = float(trainedSetLen)
    ftrashnum = float(trainedSetLen)
    for w in wordset:   #拉普拉斯平滑
        if normaldic.get(w):
            nodic[w] = np.log10((normaldic[w] + 1) / (fnormalnum + maxpahdic[w]))
        else:
            nodic[w] = np.log10( 1 / (fnormalnum + maxpahdic[w]))
        if trashdic.get(w):
            trdic[w] = np.log10((trashdic[w] + 1) / (ftrashnum + maxpahdic[w]))
        else:
            trdic[w] = np.log10( 1 / (ftrashnum + maxpahdic[w]))
    return nodic, trdic

def mailtest(test_path,ndic,tdic):
    ##测试
    normaltrash = []
    trashtrash = []
    folder_list = os.listdir(test_path)
    for fir_list in folder_list:
        #print fir_list
        if fir_list == 'normal':
            sec_folder_path = test_path + '/normal'
            sec_folder_list = os.listdir(sec_folder_path)
            for sec_list in sec_folder_list:
                thi_folder_path = sec_folder_path + '/' + sec_list
                if istrashjieba(thi_folder_path,3,ndic,tdic):
                    normaltrash.append(thi_folder_path)
        else:
            sec_folder_path = test_path + '/trash'
            sec_folder_list = os.listdir(sec_folder_path)
            for sec_list in sec_folder_list:
                thi_folder_path = sec_folder_path + '/' + sec_list
                if istrashjieba(thi_folder_path,3,ndic,tdic):
                    trashtrash.append(thi_folder_path)
    return normaltrash,trashtrash

def trainWithTest(train_path,testpath,norepath,trrepath):
    ##在垃圾邮件和正常邮件中获取出现的词及其词频
    folder_list = os.listdir(train_path)
    wordset = set()
    maxpahdic = {}

    #######################3
    plt.ion() #开启interactive mode 成功的关键函数
    fig = plt.figure(1)
    t = []
    t_now = 0
    nor_acc = []
    tra_acc = []
    #############
    
    stopword = stopwordlist('./stop_word1.txt')
    for fir_list in folder_list:
        #print fir_list
        if fir_list == 'normal':
            nor_sec_folder_path = train_path + '/normal'
            nor_sec_folder_list = os.listdir(nor_sec_folder_path)         
        elif fir_list == 'trash':
            tra_sec_folder_path = train_path + '/trash'
            tra_sec_folder_list = os.listdir(tra_sec_folder_path)
            
    normalnum = len(nor_sec_folder_list)
    trashnum = len(tra_sec_folder_list)
    list_len = min(normalnum, trashnum)

    show_list = [5,10,20,50,100,150,200,400,1000,1500,2000,3000,4000,5000,5500]
    for idx in range(0,list_len):
        nor_thi_folder_path = nor_sec_folder_path + '/' + nor_sec_folder_list[idx]
        doc = open(nor_thi_folder_path,'r',errors='ignore')
        sentence = doc.read()
        word_seg = jieba.cut(sentence,cut_all = False)
        for word in word_seg:
            if contain_zh(word) and word not in stopword:
                wordset.add(word)
                if word in normaldic:
                    normaldic[word] += 1
                else:
                    normaldic[word] = 1
        doc.close()
        ##################################################   
        tra_thi_folder_path = tra_sec_folder_path + '/' + tra_sec_folder_list[idx]
        doc = open(tra_thi_folder_path,'r',errors='ignore')
        sentence = doc.read()
        word_seg = jieba.cut(sentence,cut_all = False)
        for word in word_seg:
            if contain_zh(word) and word not in stopword:
                wordset.add(word)
                if word in trashdic:
                    trashdic[word] += 1
                else:
                    trashdic[word] = 1
        doc.close()
        ##################################################
        if idx in show_list:
            nd, td = GetTrainedDic(wordset=wordset, raw_nor_dic=normaldic, raw_tra_dic=trashdic, trainedSetLen=idx)
            NormalJudge2trash, TrashJudge2trash = mailtest(testpath, nd, td)
            normal_accuracy = 1 - len(NormalJudge2trash) / 2000.0
            trash_accuracy  = len(TrashJudge2trash) / 2000.0

            t_now = idx
            t.append(t_now)#模拟数据增量流入
            nor_acc.append(normal_accuracy)#模拟数据增量流入
            tra_acc.append(trash_accuracy)
            plt.title('The Test Accuracy When Train', fontsize=16)
            plt.plot(t,nor_acc,'-r',label='normal')
            plt.plot(t,tra_acc,'-b',label='trash')
            plt.xlabel('mails', fontsize=12)
            plt.ylabel('Accuracy', fontsize=12)
            plt.legend(['normal','trash'])

            plt.draw()#注意此函数需要调用
            plt.pause(0.005)
            plt.savefig('./test_train/%5d.png'%idx)
            print('After %4d mails were trained, the accuracy of the normal testset is %.4f'%(idx,normal_accuracy))
            print('                              the accuracy of the trash testset is %.4f'% trash_accuracy )
                            
    nodic, trdic = GetTrainedDic(wordset=wordset, raw_nor_dic=normaldic, raw_tra_dic=trashdic, trainedSetLen=list_len)
    NormalJudge2trash, TrashJudge2trash = mailtest(testpath, nd, td)
    normal_accuracy = 1 - len(NormalJudge2trash) / 2000.0
    trash_accuracy  = len(TrashJudge2trash) / 2000.0

    t_now = idx
    t.append(t_now)#模拟数据增量流入
    nor_acc.append(normal_accuracy)#模拟数据增量流入
    tra_acc.append(trash_accuracy)
    plt.title('The Test Accuracy When Train', fontsize=16)
    plt.plot(t,nor_acc,'-r',label='normal')
    plt.plot(t,tra_acc,'-b',label='trash')
    plt.xlabel('mails', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(['normal','trash'])

    plt.draw()#注意此函数需要调用
    plt.pause(0.005)
    plt.savefig('./test_train/%5d.png'%list_len)
    print('After %4d mails were trained, the accuracy of the normal testset is %.4f'%(list_len,normal_accuracy))
    print('                              the accuracy of the trash testset is %.4f'% trash_accuracy )

    create_gif('./traintest.gif', './test_train/', duration = 1)
    fnormal = open(norepath,"w",errors='ignore')
    ftrash = open(trrepath,"w",errors='ignore')
    for key in nodic:
        fnormal.write(key)
        fnormal.write(' ')
        fnormal.write(str(nodic[key]))
        fnormal.write('\n')
    for key in trdic:
        ftrash.write(key)
        ftrash.write(' ')
        ftrash.write(str(trdic[key]))
        ftrash.write('\n')
    fnormal.close()
    ftrash.close()
    return

def jbmailtrainer(train_path,norepath,trrepath):
    ##在垃圾邮件和正常邮件中获取出现的词及其词频
    folder_list = os.listdir(train_path)
    wordset = set()

    show_th = 0
    maxpahdic = {}

    stopword = stopwordlist('./stop_word1.txt')
    for fir_list in folder_list:
        #print fir_list
        if fir_list == 'normal':
            sec_folder_path = train_path + '/normal'
            sec_folder_list = os.listdir(sec_folder_path)
            normalnum = len(sec_folder_list)
            print (type(normalnum),normalnum)
            for sec_list in sec_folder_list:
                thi_folder_path = sec_folder_path + '/' + sec_list
                doc = open(thi_folder_path,'r',errors='ignore')
                sentence = doc.read()
                word_seg = jieba.cut(sentence,cut_all = False)
                temp = []
                for word in word_seg:
                    if contain_zh(word) and word not in stopword:
                        wordset.add(word)
                        temp.append(word)
                        if word in normaldic:
                            normaldic[word] += 1
                        else:
                            normaldic[word] = 1
                doc.close()
                ##################################################
                show_th += 1
                if show_th % 200 == 0:
                    idx = 0
                    print('When %d mail were trained:'%show_th)
                    for k in normaldic:
                        print('The number of the word %s is %d!'%(k,normaldic[k]))
                        idx += 1
                        if idx >= 9:
                            break
        else:
            sec_folder_path = train_path + '/trash'
            sec_folder_list = os.listdir(sec_folder_path)
            trashnum = len(sec_folder_list)
            print (type(trashnum),trashnum)
            for sec_list in sec_folder_list:
                thi_folder_path = sec_folder_path + '/' + sec_list
                doc = open(thi_folder_path,'r',errors='ignore')
                sentence = doc.read()
                word_seg = jieba.cut(sentence,cut_all = False)
                for word in word_seg:
                    if contain_zh(word) and word not in stopword:
                        wordset.add(word)
                        temp.append(word)######
                        if word in trashdic:
                            trashdic[word] += 1
                        else:
                            trashdic[word] = 1
                doc.close()
                ##################################################
                show_th += 1
                if show_th % 200 == 0:
                    idx = 0
                    print('When %d mail were trained:'%show_th)
                    for k in normaldic:
                        print('The number of the word %s is %d!'%(k,normaldic[k]))
                        idx += 1
                        if idx >= 9:
                            break
                    idx = 0
                    for k in trashdic:
                        print('The number of the word %s is %d!'%(k,trashdic[k]))
                        idx += 1
                        if idx >= 9:
                            break
                            
    ##合并得到最大频率表
    """
    stopword = stopwordlist('./stop_word1.txt')
    raw_word = copy.deepcopy(normaldic)
    for word in raw_word:
        if word in stopword:
            normaldic.pop(word,'404')
    raw_word = copy.deepcopy(trashdic)
    for word in raw_word:
        if word in stopword:
            trashdic.pop(word,'404')
    raw_word = wordset.copy()
    for word in raw_word:
        if word in stopword:
            wordset.remove(word)"""
    
    maxpahdic = {}
    for w in wordset:
        if normaldic.get(w):
            maxpahdic[w] = normaldic[w]
        else:
            maxpahdic[w] = 0
        if trashdic.get(w) and (maxpahdic[w] < trashdic[w]):
                maxpahdic[w] = trashdic[w]
    
    nodic = {}
    trdic = {}
    fnormalnum = float(normalnum)
    ftrashnum = float(trashnum)
    for w in wordset:   #拉普拉斯平滑
        if normaldic.get(w):
            nodic[w] = np.log10((normaldic[w] + 1) / (fnormalnum + maxpahdic[w]))
        else:
            nodic[w] = np.log10( 1 / (fnormalnum + maxpahdic[w]))
        if trashdic.get(w):
            trdic[w] = np.log10((trashdic[w] + 1) / (ftrashnum + maxpahdic[w]))
        else:
            trdic[w] = np.log10( 1 / (ftrashnum + maxpahdic[w]))
        
    fnormal = open(norepath,"w",errors='ignore')
    ftrash = open(trrepath,"w",errors='ignore')
    for key in nodic:
        fnormal.write(key)
        fnormal.write(' ')
        fnormal.write(str(nodic[key]))
        fnormal.write('\n')
    for key in trdic:
        ftrash.write(key)
        ftrash.write(' ')
        ftrash.write(str(trdic[key]))
        ftrash.write('\n')
    fnormal.close()
    ftrash.close()
    return nodic,trdic

def istrashjieba(textpath,T,ndic,tdic):
    ##判断是否为垃圾邮件
    doc = open(textpath,'r',errors='ignore')
    sentence = doc.read()
    doc.close()
    stopword = stopwordlist('./stop_word1.txt')
    word_seg = jieba.cut(sentence,cut_all = False)
    word_feature = []
    norp = np.log10(0.5)
    trap = np.log10(0.5)
    for word in word_seg:
        if contain_zh(word) and word not in stopword:
            word_feature.append(word)
    for fea in word_feature:
        if fea in ndic:
            norp += ndic[fea]
        if fea in tdic:
            trap += tdic[fea]
    if norp > (np.log10(T) + trap):
        return False
    else:
        return True

def jbmailtester(test_path,norepath,trrepath):
    ##测试
    normaltrash = []
    trashtrash = []
    ndic,tdic = getTrainset(norepath,trrepath)
    folder_list = os.listdir(test_path)

    #############
    normalnum = 0.0;
    trashnum  = 0.0;

    NormalJudge2trash = 0;
    TrashJudge2trash = 0;

    x_list = []
    nor_list = []
    tra_list = []

    plt.ion() #开启interactive mode 成功的关键函数
    fig = plt.figure(1)
    t = []
    t_now = 0
    nor_acc = []
    tra_acc = []

    #############

    for fir_list in folder_list:
        #print fir_list
        if fir_list == 'normal':
            nor_sec_folder_path = test_path + '/normal'
            nor_sec_folder_list = os.listdir(nor_sec_folder_path)

        else:
            tra_sec_folder_path = test_path + '/trash'
            tra_sec_folder_list = os.listdir(tra_sec_folder_path)
            
    list_len = min(len(nor_sec_folder_list),len(tra_sec_folder_list))

    for idx in range(0,list_len):
        nor_thi_folder_path = nor_sec_folder_path + '/' + nor_sec_folder_list[idx]
        tra_thi_folder_path = tra_sec_folder_path + '/' + tra_sec_folder_list[idx]

        if istrashjieba(nor_thi_folder_path,3,ndic,tdic):
            normaltrash.append(nor_thi_folder_path)
            NormalJudge2trash += 1
        normalnum += 1

        if istrashjieba(tra_thi_folder_path,3,ndic,tdic):
            trashtrash.append(tra_thi_folder_path)
            TrashJudge2trash += 1
        trashnum += 1

        #ax = fig.add_subplot()
        if idx % 50 == 25:
            normal_accuracy = 1 - NormalJudge2trash/normalnum
            trash_accuracy  = TrashJudge2trash/trashnum
            print('After %d normal mail was tested, the accuracy is %.4f'%(idx, normal_accuracy))
            x_list.append(idx)
            nor_list.append(normal_accuracy)
            print('After %d trash mail was tested, the accuracy is %.4f'%(idx, trash_accuracy))
            tra_list.append(trash_accuracy)


            t_now = idx
            t.append(t_now)#模拟数据增量流入
            nor_acc.append(normal_accuracy)#模拟数据增量流入
            tra_acc.append(trash_accuracy)
            plt.title('The Test Accuracy', fontsize=16)
            plt.plot(t,nor_acc,'-r',label='normal')
            plt.plot(t,tra_acc,'-b',label='trash')
            """for x, y in zip(t, nor_acc):
                                                    print(x ,y)
                                                    plt.text(x, y+0.01, '%.4f'%y, ha='center', va='bottom', fontsize=10)
                                                for x, y in zip(t, tra_acc):
                                                    plt.text(x, y+0.01, '%.4f'%y, ha='center', va='bottom', fontsize=10)"""
            plt.xlabel('mails', fontsize=12)
            plt.ylabel('Accuracy', fontsize=12)
            plt.legend(['normal','trash'])

            plt.draw()#注意此函数需要调用
            plt.pause(0.005)
            plt.savefig('./test_process/%5d.png'%idx)
            

    normal_accuracy = 1 - NormalJudge2trash/normalnum
    trash_accuracy  = TrashJudge2trash/trashnum
    print('After %d normal mail was tested, the accuracy is %.4f'%(len(nor_sec_folder_list), normal_accuracy))
    print('After %d trash mail was tested, the accuracy is %.4f'%(len(tra_sec_folder_list), trash_accuracy))
    x_list.append(normalnum)
    nor_list.append(normal_accuracy)
    tra_list.append(trash_accuracy )

    t_now = list_len
    t.append(t_now)#模拟数据增量流入
    nor_acc.append(normal_accuracy)#模拟数据增量流入
    tra_acc.append(trash_accuracy)
    plt.title('The Test Accuracy', fontsize=16)
    plt.plot(t,nor_acc,'-r',label='normal')
    plt.plot(t,tra_acc,'-b',label='trash')
    plt.xlabel('mails', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(['normal','trash'])
    """for x, y in zip(t, nor_acc):
                    plt.text(x, y+0.01, '%.4f'%y, ha='center', va='bottom', fontsize=10)
                for x, y in zip(t, tra_acc):
                    plt.text(x, y+0.01, '%.4f'%y, ha='center', va='bottom', fontsize=10)"""
    plt.draw()#注意此函数需要调用
    plt.pause(0.005)
    plt.savefig('./test_process/%5d.png'%t_now)

    create_gif('./testall.gif', './test_process/', duration = 1)

    draw(x_list, nor_list, tra_list, './test_result')
    return normaltrash,trashtrash

def jbmailtesterT(test_path,norepath,trrepath,T):
    ##测试
    normaltrash = []
    trashtrash = []
    ndic,tdic = getTrainset(norepath,trrepath)
    folder_list = os.listdir(test_path)
    for fir_list in folder_list:
        #print fir_list
        if fir_list == 'normal':
            sec_folder_path = test_path + '/normal'
            sec_folder_list = os.listdir(sec_folder_path)
            for sec_list in sec_folder_list:
                thi_folder_path = sec_folder_path + '/' + sec_list
                if istrashjieba(thi_folder_path,T,ndic,tdic):
                    normaltrash.append(thi_folder_path)
        else:
            sec_folder_path = test_path + '/trash'
            sec_folder_list = os.listdir(sec_folder_path)
            for sec_list in sec_folder_list:
                thi_folder_path = sec_folder_path + '/' + sec_list
                if istrashjieba(thi_folder_path,T,ndic,tdic):
                    trashtrash.append(thi_folder_path)
    return normaltrash,trashtrash

def testT(test_path,norepath,trrepath,minT,maxT):
    Tlist = [ele for ele in range(minT,maxT+1)] 
    list_pN = []
    list_pT = []

    for th in Tlist:
        n, t = jbmailtesterT(test_path,norepath,trrepath,th)
        list_pN.append(1.0-len(n)/2000.0)
        list_pT.append(len(t)/2000.0)
        print('When T = %d\n \
            The accuracy of the normal mail is %.4f.\n \
            The accuracy of the trash mail is %.4f.'% \
            (th,1.0-len(n)/2000.0,len(t)/2000.0))
    writelist(list_pN,'./list_pN.txt')
    writelist(list_pT,'./list_pT.txt')

def testT10(test_path,norepath,trrepath,minT,maxT):
    Tlist = [pow(10,ele) for ele in range(minT,maxT+1)] 
    list_pN = []
    list_pT = []

    plt.ion() #开启interactive mode 成功的关键函数
    fig = plt.figure(1)
    t = []
    t_now = 0
    nor_acc = []
    tra_acc = []
    count = minT
    for th in Tlist:
        nor, tra = jbmailtesterT(test_path,norepath,trrepath,float(th))
        normal_accuracy = 1.0-len(nor)/2000.0
        trash_accuracy  = len(tra)/2000.0
        print('When T = 10**%d\n \
            The accuracy of the normal mail is %.4f.\n \
            The accuracy of the trash mail is %.4f.'% \
            (count,normal_accuracy,trash_accuracy))
        t_now = count
        t.append(t_now)#模拟数据增量流入
        nor_acc.append(normal_accuracy)#模拟数据增量流入
        tra_acc.append(trash_accuracy)
        plt.title('The Test Accuracy', fontsize=16)
        plt.plot(t,nor_acc,'-r',label='normal')
        plt.plot(t,tra_acc,'-b',label='trash')
        plt.xlabel('T', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.legend(['normal','trash'])

        plt.draw()#注意此函数需要调用
        plt.pause(0.005)
        plt.savefig('./test_T/%5d.png'%count)
        count += 1
    create_gif('./testT.gif', './test_T/', duration = 1)

def P_of_mail(textpath,T,ndic,tdic):
    ##判断是否为垃圾邮件
    ####################
    plt.ion() #开启interactive mode 成功的关键函数
    fig = plt.figure(1)
    t = []
    t_now = 0
    nor_psb = []
    tra_psb = []
    #####################

    doc = open(textpath,'r',errors='ignore')
    sentence = doc.read()
    doc.close()
    stopword = stopwordlist('./stop_word1.txt')
    word_seg = jieba.cut(sentence,cut_all = False)
    word_feature = []
    norp = 0.5
    trap = 0.5
    for word in word_seg:
        if contain_zh(word) and word not in stopword:
            word_feature.append(word)
    #print(word_feature)
    count = 0
    for fea in word_feature:
        if fea in ndic:
            norp += ndic[fea]
        if fea in tdic:
            trap += tdic[fea]
        count += 1
        if count % 2 == 0:
            nor = 1.0/(1+pow(10,trap-norp))
            tra = 1.0/(1+pow(10,norp-trap))
            print('After %3d feature word, the normal posibility of the mail is %.4f'%(count,nor))
            print('                        the trash posibility of the mail is %.4f'%tra)

            t_now = count
            t.append(t_now)#模拟数据增量流入
            nor_psb.append(nor)#模拟数据增量流入
            tra_psb.append(tra)
            plt.title('The Test Posibility', fontsize=16)
            plt.plot(t,nor_psb,'-r',label='normal')
            plt.plot(t,tra_psb,'-b',label='trash')
            plt.xlabel('features', fontsize=12)
            plt.ylabel('posibility', fontsize=12)
            plt.legend(['normal','trash'])

            plt.draw()#注意此函数需要调用
            plt.pause(0.005)
            plt.savefig('./test_one/%5d.png'%count)

    nor = 1.0/(1+pow(10,trap-norp))
    tra = 1.0/(1+pow(10,norp-trap))
    print('After %3d feature word, the normal posibility of the mail is %.4f'%(count,nor))
    print('                        the trash posibility of the mail is %.4f'%tra)
    t_now = len(word_feature)
    t.append(t_now)#模拟数据增量流入
    nor_psb.append(nor)#模拟数据增量流入
    tra_psb.append(tra)
    plt.title('The Test Posibility', fontsize=16)
    plt.plot(t,nor_psb,'-r',label='normal')
    plt.plot(t,tra_psb,'-b',label='trash')
    plt.xlabel('features', fontsize=12)
    plt.ylabel('posibility', fontsize=12)
    plt.legend(['normal','trash'])

    plt.draw()#注意此函数需要调用
    plt.pause(0.005)
    plt.savefig('./test_one/%5d.png'% len(word_feature) )

    create_gif('./testone.gif', './test_one/', duration = 1)
    return nor,tra

def testPNG(pngpath, T, ndic, tdic):
    starttime = datetime.datetime.now()
    image = Image.open(pngpath)
    text = pytesseract.image_to_string(image, lang='chi_sim')  # 使用简体中文解析图片
    endtime = datetime.datetime.now()
    print ('转换完成，耗时：' + str((endtime - starttime).seconds))

    text=text.replace(' ','')
    text=text.replace('\n','')
    
    stopword = stopwordlist('./stop_word1.txt')
    word_seg = jieba.cut(text,cut_all = False)
    word_feature = []
    norp = 0.5
    trap = 0.5
    for word in word_seg:
        if contain_zh(word) and word not in stopword:
            word_feature.append(word)
    print(word_feature)
    for fea in word_feature:
        if fea in ndic:
            norp += ndic[fea]
        if fea in tdic:
            trap += tdic[fea]

    nor = 1.0/(1+pow(10,trap-norp))
    tra = 1.0/(1+pow(10,norp-trap))
    return nor,tra       

def main():
    trainpath = './train'
    testpath = './test'

    jbnore_path = "normalmail1.txt"
    jbtrre_path = "trashmail1.txt"

    nore_path = "normalmail2.txt"
    trre_path = "trashmail2.txt"

    mode_type = ['train', 'tstall','tstone','tstT']
    mode = str(sys.argv[2])

    print('The mode is: %s'%mode)
    if mode == 'train':         #训练分类器
        #python mailjb.py -mode train
        _, _ = jbmailtrainer(trainpath,jbnore_path,jbtrre_path)
    elif mode == 'traintest':
        trainWithTest(trainpath,testpath,nore_path,trre_path) 
    elif mode == 'tstall':      #测试分类器
        #python mailjb.py -mode tstall
        n,t = jbmailtester(testpath,jbnore_path,jbtrre_path)
        print(len(n),len(t))
        print ('the accuracy of the normal mail is: %.4f'%(1.0-len(n)/2000.0))
        print ('the accuracy of the trash mail is: %.4f'%(len(t)/2000.0))
    elif mode == 'tstone':      #测试指定文件
        #python mailjb.py -mode tstone (待分类文件路径)
        filepath = str(sys.argv[3])
        ndic,tdic = getTrainset(jbnore_path,jbtrre_path)
        n,t = P_of_mail(filepath,1,ndic,tdic)
        print('The posibility that the mail is normal is alomost %.4f!'%n)
        print('The posibility that the mail is trash is nearly %.4f!'%t)
    elif mode == 'tstT':        #测试阈值的影响
        #python mailjb.py -mode tstT /min /max min为阈值下界，max为阈值上界
        minT = int(sys.argv[3])
        maxT = int(sys.argv[4])
        testT(testpath,jbnore_path,jbtrre_path,minT,maxT)
    elif mode == 'tstT10':        #测试阈值的影响
        #python mailjb.py -mode tstT /min /max min为阈值下界，max为阈值上界
        minT = int(sys.argv[3])
        maxT = int(sys.argv[4])
        testT10(testpath,jbnore_path,jbtrre_path,minT,maxT)
    elif mode == 'tstpng':
        filepath = str(sys.argv[3])
        ndic,tdic = getTrainset(jbnore_path,jbtrre_path)
        n, t = testPNG(filepath,1,ndic,tdic)
        print('The posibility that the mail is normal is alomost %.4of!'%n)
        print('The posibility that the mail is trash is nearly %.4f!'%t)
    else:
        print('The mode \'%s\' doesn\'t exist!')

#print(stopwordlist('./stop_word1.txt'))
main()