#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
@AUTHOR:Joselyn
@CONTACT:zhaojing17@foxmail.com
@SOFTWERE:PyCharm
@FILE:web01.py
@TIME:2018/11/11 13:28
@DES:
'''

import os
import jieba
import re
import numpy as np

import sys

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



trainpath = './train'
testpath = './test'

jbnore_path = "normalmail1.txt"
jbtrre_path = "trashmail1.txt"

trainall_file = "trainall_out.txt"

testall_file = 'testall_out.txt'

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

def stopwordlist(filepath):
    lis = []
    doc = open(filepath,'r')
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


def jbmailtrainer(train_path, norepath, trrepath):
    ##在垃圾邮件和正常邮件中获取出现的词及其词频
    folder_list = os.listdir(train_path)
    wordset = set()
    maxpahdic = {}

    word_num = 0  # zj

    # # 将终端输出指向文件
    # output = sys.stdout
    # outputfile = open(trainall_file, 'w')
    # sys.stdout = outputfile

    for fir_list in folder_list:
        # print fir_list
        if fir_list == 'normal':
            sec_folder_path = train_path + '/normal'
            sec_folder_list = os.listdir(sec_folder_path)
            normalnum = len(sec_folder_list)

            print (type(normalnum), normalnum)



            for sec_list in sec_folder_list:
                thi_folder_path = sec_folder_path + '/' + sec_list
                doc = open(thi_folder_path, 'r')
                sentence = doc.read()
                word_seg = jieba.cut(sentence, cut_all=False)
                for word in word_seg:
                    # print type(word)
                    if contain_zh(word):
                        # print word
                        wordset.add(word)

                        # 强行终止
                        if (word_num > 1000):
                            break

                        if word in normaldic:
                            normaldic[word] += 1
                            word_num = word_num + 1



                            print (str(word_num) + " :In normaldic " + word.encode('utf8') + " has " + str(normaldic[word]))
                        else:
                            normaldic[word] = 1
                            # zj
                            word_num=word_num+1
                            print (str(word_num)+" : normaldic 中添加新词 --" +word.encode('utf8'))
                doc.close()

        else:
            sec_folder_path = train_path + '/trash'
            sec_folder_list = os.listdir(sec_folder_path)
            trashnum = len(sec_folder_list)
            print (type(trashnum), trashnum)

            # word_num=0

            for sec_list in sec_folder_list:
                thi_folder_path = sec_folder_path + '/' + sec_list
                doc = open(thi_folder_path, 'r')
                sentence = doc.read()
                word_seg = jieba.cut(sentence, cut_all=False)
                for word in word_seg:
                    if contain_zh(word) :
                        wordset.add(word)

                        # 强行终止
                        if (word_num > 1000):
                            break

                        if word in trashdic:
                            trashdic[word] += 1
                            word_num = word_num + 1
                            print (str(word_num)+" :In trashdic "+word.encode('utf8')+" has "+str(trashdic[word]))
                        else:
                            trashdic[word] = 1
                            #zj
                            word_num = word_num + 1
                            print (str(word_num) + " : trashdic 中添加新词 --" + word.encode('utf8'))
                doc.close()
    #  # 将终端输出修改回来
    # outputfile.close()
    # sys.stdout = output

    ##合并得到最大频率表


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
    for w in wordset:  # 拉普拉斯平滑
        # nodic[w] = np.log10((normaldic[w] + 1) / (fnormalnum + maxpahdic[w]))
        if normaldic.get(w):
            nodic[w] = np.log10((normaldic[w] + 1) / (fnormalnum + 2))
            # nodic[w] = np.log10((normaldic[w] + maxpahdic[w]) / (fnormalnum + 1))
        else:
            nodic[w] = np.log10((maxpahdic[w]) / (fnormalnum + 1))

        # trdic[w] = np.log10((trashdic[w] + 1) / (ftrashnum + maxpahdic[w]))
        if trashdic.get(w):
            trdic[w] = np.log10((trashdic[w] + 1) / (ftrashnum + 2))
            # trdic[w] = np.log10((trashdic[w] + maxpahdic[w]) / (ftrashnum + 1))
        else:
            trdic[w] = np.log10((maxpahdic[w]) / (ftrashnum + 1))

    fnormal = open(norepath, "w")
    ftrash = open(trrepath, "w")
    for key in nodic:
        fnormal.write(key.encode('utf8'))
        fnormal.write(' ')
        fnormal.write(str(nodic[key]))
        fnormal.write('\n')
    for key in trdic:
        ftrash.write(key.encode('utf8'))
        ftrash.write(' ')
        ftrash.write(str(trdic[key]))
        ftrash.write('\n')
    fnormal.close()
    ftrash.close()
    return nodic, trdic

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

def trainWithTest(train_path, testpath, norepath, trrepath):
    ##在垃圾邮件和正常邮件中获取出现的词及其词频
    folder_list = os.listdir(train_path)
    wordset = set()
    maxpahdic = {}

    #######################3
    plt.ion()  # 开启interactive mode 成功的关键函数
    fig = plt.figure(1)
    t = []
    t_now = 0
    nor_acc = []
    tra_acc = []
    #############

    stopword = stopwordlist('./stop_word1.txt')
    for fir_list in folder_list:
        # print fir_list
        if fir_list == 'normal':
            nor_sec_folder_path = train_path + '/normal'
            nor_sec_folder_list = os.listdir(nor_sec_folder_path)
        elif fir_list == 'trash':
            tra_sec_folder_path = train_path + '/trash'
            tra_sec_folder_list = os.listdir(tra_sec_folder_path)

    normalnum = len(nor_sec_folder_list)
    trashnum = len(tra_sec_folder_list)
    list_len = min(normalnum, trashnum)

    show_list = [5, 10, 20, 50, 100, 150, 200, 400, 1000, 1500, 2000, 3000, 4000, 5000, 5500]
    for idx in range(0, list_len):
        nor_thi_folder_path = nor_sec_folder_path + '/' + nor_sec_folder_list[idx]
        doc = open(nor_thi_folder_path, 'r', errors='ignore')
        sentence = doc.read()
        word_seg = jieba.cut(sentence, cut_all=False)
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
        doc = open(tra_thi_folder_path, 'r', errors='ignore')
        sentence = doc.read()
        word_seg = jieba.cut(sentence, cut_all=False)
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
            trash_accuracy = len(TrashJudge2trash) / 2000.0

            t_now = idx
            t.append(t_now)  # 模拟数据增量流入
            nor_acc.append(normal_accuracy)  # 模拟数据增量流入
            tra_acc.append(trash_accuracy)
            plt.title('The Test Accuracy When Train', fontsize=16)
            plt.plot(t, nor_acc, '-r', label='normal')
            plt.plot(t, tra_acc, '-b', label='trash')
            plt.xlabel('mails', fontsize=12)
            plt.ylabel('Accuracy', fontsize=12)
            plt.legend(['normal', 'trash'])

            plt.draw()  # 注意此函数需要调用
            plt.pause(0.005)
            plt.savefig('./test_train/%5d.png' % idx)
            print('After %4d mails were trained, the accuracy of the normal testset is %.4f' % (idx, normal_accuracy))
            print('                              the accuracy of the trash testset is %.4f' % trash_accuracy)

    nodic, trdic = GetTrainedDic(wordset=wordset, raw_nor_dic=normaldic, raw_tra_dic=trashdic, trainedSetLen=list_len)
    NormalJudge2trash, TrashJudge2trash = mailtest(testpath, nd, td)
    normal_accuracy = 1 - len(NormalJudge2trash) / 2000.0
    trash_accuracy = len(TrashJudge2trash) / 2000.0

    t_now = idx
    t.append(t_now)  # 模拟数据增量流入
    nor_acc.append(normal_accuracy)  # 模拟数据增量流入
    tra_acc.append(trash_accuracy)
    plt.title('The Test Accuracy When Train', fontsize=16)
    plt.plot(t, nor_acc, '-r', label='normal')
    plt.plot(t, tra_acc, '-b', label='trash')
    plt.xlabel('mails', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(['normal', 'trash'])

    plt.draw()  # 注意此函数需要调用
    plt.pause(0.005)
    plt.savefig('./test_train/%5d.png' % list_len)
    print('After %4d mails were trained, the accuracy of the normal testset is %.4f' % (list_len, normal_accuracy))
    print('                              the accuracy of the trash testset is %.4f' % trash_accuracy)

    create_gif('./traintest.gif', './test_train/', duration=1)
    fnormal = open(norepath, "w", errors='ignore')
    ftrash = open(trrepath, "w", errors='ignore')
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


def jbmailtrainer_zj(train_path, norepath, trrepath):
    ##在垃圾邮件和正常邮件中获取出现的词及其词频
    folder_list = os.listdir(train_path)
    wordset = set()
    maxpahdic = {}

    word_num = 0  # zj

    # 将终端输出指向文件
    output = sys.stdout
    outputfile = open(trainall_file, 'w')
    # outputfile.write(" ")
    sys.stdout = outputfile

    for fir_list in folder_list:
        # print fir_list
        if fir_list == 'normal':
            sec_folder_path = train_path + '/normal'
            sec_folder_list = os.listdir(sec_folder_path)
            normalnum = len(sec_folder_list)

            # print (type(normalnum), normalnum)



            for sec_list in sec_folder_list:
                thi_folder_path = sec_folder_path + '/' + sec_list
                doc = open(thi_folder_path, 'r')
                sentence = doc.read()
                word_seg = jieba.cut(sentence, cut_all=False)
                for word in word_seg:
                    # print type(word)
                    if contain_zh(word):
                        # print word
                        wordset.add(word)

                        # # 强行终止
                        # if (word_num > 1000):
                        #     break

                        if word in normaldic:
                            normaldic[word] += 1
                            # word_num = word_num + 1



                            # print (str(word_num) + " :In normaldic " + word.encode('utf8') + " has " + str(normaldic[word]))
                        else:
                            normaldic[word] = 1
                            # zj
                            # word_num=word_num+1
                            print (" normaldic 中添加新词 --" +word.encode('utf8'))
                doc.close()

        else:
            sec_folder_path = train_path + '/trash'
            sec_folder_list = os.listdir(sec_folder_path)
            trashnum = len(sec_folder_list)
            # print (type(trashnum), trashnum)

            # word_num=0

            for sec_list in sec_folder_list:
                thi_folder_path = sec_folder_path + '/' + sec_list
                doc = open(thi_folder_path, 'r')
                sentence = doc.read()
                word_seg = jieba.cut(sentence, cut_all=False)
                for word in word_seg:
                    if contain_zh(word) :
                        wordset.add(word)

                        # # 强行终止
                        # if (word_num > 1000):
                        #     break

                        if word in trashdic:
                            trashdic[word] += 1
                            # word_num = word_num + 1
                            # print (str(word_num)+" :In trashdic "+word.encode('utf8')+" has "+str(trashdic[word]))
                        else:
                            normaldic[word] = 1
                            #zj
                            # word_num = word_num + 1
                            print ( " trashdic 中添加新词 --" + word.encode('utf8'))
                doc.close()
     # 将终端输出修改回来
    outputfile.close()
    sys.stdout = output

    print "输出完毕"
    return normaldic,normaldic

def istrashjieba(textpath,T,ndic,tdic):
    ##判断是否为垃圾邮件
    doc = open(textpath,'r')
    sentence = doc.read()
    doc.close()
    word_seg = jieba.cut(sentence,cut_all = False)
    word_feature = []
    norp = np.log10(0.5)
    trap = np.log10(0.5)
    for word in word_seg:

        if contain_zh(word):
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

# def jbmailtester(test_path,norepath,trrepath):
#     ##测试
#     normaltrash = []
#     trashtrash = []
#     ndic,tdic = getTrainset(norepath,trrepath)
#     folder_list = os.listdir(test_path)
#     for fir_list in folder_list:
#         #print fir_list
#         if fir_list == 'normal':
#             sec_folder_path = test_path + '/normal'
#             sec_folder_list = os.listdir(sec_folder_path)
#             for sec_list in sec_folder_list:
#                 thi_folder_path = sec_folder_path + '/' + sec_list
#                 if istrashjieba(thi_folder_path,3,ndic,tdic):
#                     normaltrash.append(thi_folder_path)
#         else:
#             sec_folder_path = test_path + '/trash'
#             sec_folder_list = os.listdir(sec_folder_path)
#             for sec_list in sec_folder_list:
#                 thi_folder_path = sec_folder_path + '/' + sec_list
#                 if istrashjieba(thi_folder_path,3,ndic,tdic):
#                     trashtrash.append(thi_folder_path)
#     return normaltrash,trashtrash

def jbmailtester_zj(test_path,norepath,trrepath):
    ##测试
    normaltrash = []
    trashtrash = []
    ndic,tdic = getTrainset(norepath,trrepath)
    folder_list = os.listdir(test_path)
    test_num = 0

    # 将终端输出指向文件
    output = sys.stdout
    outputfile = open(testall_file, 'w')
    outputfile.write(" ")
    sys.stdout = outputfile

    for fir_list in folder_list:
        #print fir_list
        if fir_list == 'normal':
            sec_folder_path = test_path + '/normal'
            sec_folder_list = os.listdir(sec_folder_path)
            print len(sec_folder_list)
            for sec_list in sec_folder_list:
                thi_folder_path = sec_folder_path + '/' + sec_list
                test_num = test_num+1

                #强行终止
                if(test_num > 1000):
                    break

                if istrashjieba(thi_folder_path,3,ndic,tdic):
                    print (str(test_num)+" :邮件 " + thi_folder_path + " 的检测结果为 --垃圾邮件--")
                    normaltrash.append(thi_folder_path)
                else:
                    print (str(test_num)+" ：邮件" + thi_folder_path + " 的检测结果为 --正常邮件--")
        else:
            sec_folder_path = test_path + '/trash'
            sec_folder_list = os.listdir(sec_folder_path)
            print len(sec_folder_list)
            for sec_list in sec_folder_list:
                thi_folder_path = sec_folder_path + '/' + sec_list
                test_num = test_num+1

                # 强行终止
                if (test_num > 1000):
                    break

                if istrashjieba(thi_folder_path,3,ndic,tdic):
                    print (str(test_num)+" :邮件 " + thi_folder_path + " 的检测结果为 --垃圾邮件--")
                    trashtrash.append(thi_folder_path)
                else:
                    print (str(test_num)+" ：邮件" + thi_folder_path + " 的检测结果为 --正常邮件--")
    # 将终端输出修改回来
    outputfile.close()
    sys.stdout = output
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





def P_of_mail(textpath,ndic,tdic):
    ##判断是否为垃圾邮件
    doc = open(textpath,'r')
    sentence = doc.read()
    doc.close()
    # stopword = stopwordlist('.stop_word1.txt')
    word_seg = jieba.cut(sentence,cut_all = False)
    word_feature = []
    norp = 0.5
    trap = 0.5
    for word in word_seg:
        if contain_zh(word):
            word_feature.append(word)
    for fea in word_feature:
        if fea in ndic:
            norp += ndic[fea]
        if fea in tdic:
            trap += tdic[fea]

    nor = 1.0/(1+pow(10,trap-norp))
    tra = 1.0/(1+pow(10,norp-trap))
    return nor,tra

def check_one_mail(mail_type,num):
    if mail_type==1:
        testpath='./test/normal'
    else:
        testpath='./test/trash'

    thi_folder_path = testpath+ '/' + str(num)
    ndic, tdic = getTrainset(jbnore_path, jbtrre_path)
    nor,tra=P_of_mail(thi_folder_path,ndic,tdic)
    return nor,tra



def get_mail_content(mail_type,num):
    if mail_type==1:
        testpath='./test/normal'
    else:
        testpath='./test/trash'

    thi_folder_path = testpath+ '/' + str(num)
    doc = open(thi_folder_path, 'r')
    sentence = doc.read()
    sentence = sentence.decode('gbk')
    doc.close()

    return sentence





from flask import request,jsonify,Response
from flask_cors import *
from flask import Flask



import random



import jieba
import jieba.analyse

# from cws_support import get_chars,get_datas

app = Flask(__name__)
CORS(app, supports_credentials=True)
ctx=app.app_context()
ctx.push()
# '''设定全局通用变量'''
global mail_type,num

@app.route('/getmail',methods=['POST','GET'])
def get_mail():
    global mail_type,num
    mail_type = random.randint(1, 2)
    num=random.randint(6001, 8000)
    content = get_mail_content(mail_type,num)
    return jsonify({'content':content,
                     'mail_type':mail_type,
                    'num':num})

@app.route('/checkmail',methods=['POST','GET'])
def chack_mail():
    global mail_type,num
    nt,tt = check_one_mail(mail_type,num)
    if nt >tt:
        flag = 1
    else:
        flag = 0
    return jsonify({'nt':nt,
                    'tt':tt,
                    'flag':flag})

@app.route('/testAll',methods=['POST','GET'])
def testAll_mail():
    n,t = jbmailtester_zj(testpath,jbnore_path,jbtrre_path)
    # print(n)
    corract_rate_for_normal = 1.0 - len(n) / 2000.0
    corract_rate_for_trash = len(t) / 2000.0
    # 涉及到写文件和读文件
    return jsonify({'result': 1,
                    'rate_normal':corract_rate_for_normal,
                    'rate_trash':corract_rate_for_trash})

@app.route('/trainAll',methods=['POST','GET'])
def trainAll_mail():
    n,t=jbmailtrainer_zj(trainpath,jbnore_path,jbtrre_path)
    if len(n)!=0 and len(t)!=0:
        result = 1
    else:
        result = 0
    return jsonify({'result':result})

@app.route('/get_file_log',methods=['POST','GET'])
def getFileinLog():
    file_name = request.args.get('file_name', '')
    doc = open(file_name, 'r')
    sentence = doc.read()
    # sentence = sentence.decode('gbk')
    doc.close()
    return jsonify({'content':sentence})





if __name__ == '__main__':
    # from functions_pre import *
    # from use_model import *

    # _, _ = jbmailtrainer_zj(trainpath, jbnore_path, jbtrre_path)
    # n,t=jbmailtester_zj(testpath,jbnore_path,jbtrre_path)
    # print(n)
    # print len(n),len(t)
    # print ('the accuracy of the normal mail is: %.4f' % (1.0 - len(n) / 2000.0))
    # print ('the accuracy of the trash mail is: %.4f' % (len(t) / 2000.0))
    app.run()