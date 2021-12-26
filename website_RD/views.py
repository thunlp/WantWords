import torch, gc, json, os, thulac, string, re, requests, hashlib, urllib.parse 
import numpy as np
from django.shortcuts import render, render_to_response
from django.http import HttpResponse
from datetime import datetime
from pytorch_transformers import *

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=6, n_jobs=1, random_state=0, init='k-means++', max_iter=10)

def md5(str):
    m = hashlib.md5()
    m.update(str.encode("utf8"))
    return m.hexdigest()
appid = '20***************79' 
secretKey = 'D2u0***********Yhz5' 

BASE_DIR = './website_RD/'
device = torch.device('cpu')
torch.backends.cudnn.benchmark = True
words_t = torch.tensor(np.array([0]))
itemsPerCol = 20
GET_NUM = 100
NUM_RESPONSE = 500
words_t = torch.tensor(np.array([0]))

tokenizer_class = BertTokenizer
tokenizer_Ch = tokenizer_class.from_pretrained('bert-base-chinese')
tokenizer_En = tokenizer_class.from_pretrained('bert-base-uncased')
#========================ChineseRD
MODE = 'Psc'
lac = thulac.thulac()

def load_data():
    (word2index, index2word, _, _, _, _, _) = np.load(BASE_DIR + 'data_inUse1.npy', allow_pickle=True)
    wd_charas = np.load(BASE_DIR + 'data_inUse2.npy', allow_pickle=True)
    ((_, _, _, wd_sems, wd_POSs),(_, mask_s)) = np.load(BASE_DIR + 'data_inUse3.npy', allow_pickle=True)
    mask_s = torch.from_numpy(mask_s).to(device)
    wd_POSs = torch.from_numpy(wd_POSs).float().to(device)
    wd_charas = torch.from_numpy(wd_charas).float().to(device)
    wd_sems = torch.from_numpy(wd_sems).float().to(device)
    wd_C = []
    mask_c = []
    mask_s = mask_s.float()
    return word2index, index2word, (wd_C, wd_sems, wd_POSs, wd_charas), (mask_c, mask_s)

word2index, index2word, wd_features, mask_ = load_data()
(wd_C, wd_sems, wd_POSs, wd_charas) = wd_features
(mask_c, mask_s) = mask_
index2word = np.array(index2word)

# 添加同义词词林用于描述为一个词时的同义词推荐
index2synset = [[] for i in range(len(word2index))]
for line in open(BASE_DIR + 'word2synset_synset.txt').readlines():
    wd = line.split()[0]
    synset = line.split()[1:]
    for syn in synset:
        index2synset[word2index[wd]].append(word2index[syn])

MODEL_FILE = BASE_DIR + 'Zh.model'
model = torch.load(MODEL_FILE, map_location=lambda storage, loc: storage)
model.eval()
wd_data_ = json.load(open(BASE_DIR+'wd_def_for_website_xhzd+ch+xh.json'))

#wd_data = dict() # 这样竟然会导致内存泄露？！因为用这个方法时内存直接耗没了，而用下面的copy方法就没问题。
wd_data = wd_data_.copy()
wd_defi = wd_data_.copy()
for wd in wd_data_:
    #wd_data[wd] = {'w': wd_data_[wd]['word'], 'd': wd_data_[wd]['definition'], 'P': wd_data_[wd]['POS'], 'l': wd_data_[wd]['length'], 'b': wd_data_[wd]['bihuashu'], 'B': wd_data_[wd]['bihuashu1st'], 'p': wd_data_[wd]['pinyin'], 's': wd_data_[wd]['pinyinshouzimu'], 'r': wd_data_[wd]['rhyme']}
    wd_data[wd] = {'w': wd_data_[wd]['word'], 'P': wd_data_[wd]['POS'], 'l': wd_data_[wd]['length'], 'b': wd_data_[wd]['bihuashu'], 'B': wd_data_[wd]['bihuashu1st'], 'p': wd_data_[wd]['pinyin'], 's': wd_data_[wd]['pinyinshouzimu'], 'r': wd_data_[wd]['rhyme']}
    wd_defi[wd] = wd_data_[wd]['definition']
del wd_data_

#========================EnglishRD
MODE_en = 'rsl'
MODEL_FILE_en = BASE_DIR + 'En.model'
wd_data_en_ = json.load(open(BASE_DIR+'wd_def_for_website_En.json'))

wd_data_en = wd_data_en_.copy()
wd_defi_en = wd_data_en_.copy()
for wd in wd_data_en_:
    #wd_data_en[wd] = {'w': wd_data_en_[wd]['word'], 'd': wd_data_en_[wd]['definition'], 'P': wd_data_en_[wd]['POS']}
    wd_data_en[wd] = {'w': wd_data_en_[wd]['word'], 'P': wd_data_en_[wd]['POS']}
    wd_defi_en[wd] = wd_data_en_[wd]['definition']
del wd_data_en_
gc.collect()

def label_multihot(labels, num):
    sm = np.zeros((len(labels), num), dtype=np.float32)
    for i in range(len(labels)):
        for s in labels[i]:
            if s >= num:
                break
            sm[i, s] = 1
    return sm

def word2feature(dataset, word_num, feature_num, feature_name):
    max_feature_num = max([len(instance[feature_name]) for instance in dataset])
    ret = np.zeros((word_num, max_feature_num), dtype=np.int64)
    ret.fill(feature_num)
    for instance in dataset:
        if ret[instance['word'], 0] != feature_num: 
            continue # this target_words has been given a feature mapping, because same word with different definition in dataset
        feature = instance[feature_name]
        ret[instance['word'], :len(feature)] = np.array(feature)
    return torch.tensor(ret, dtype=torch.int64, device=device)
    
def mask_noFeature(label_size, wd2fea, feature_num):
    mask_nofea = torch.zeros(label_size, dtype=torch.float32, device=device)
    for i in range(label_size):
        feas = set(wd2fea[i].detach().cpu().numpy().tolist())-set([feature_num])
        if len(feas)==0:
            mask_nofea[i] = 1
    return mask_nofea
 
(_, (_, label_size, _, _), (word2index_en, index2word_en, index2sememe, index2lexname, index2rootaffix)) = np.load(BASE_DIR + 'data_inUse1_en.npy', allow_pickle=True)
print('-------------------------1')   
#(data_train_idx, data_dev_idx, data_test_idx) = np.load(BASE_DIR + 'data_inUse2_en.npy', allow_pickle=True)
#data_all_idx = data_train_idx + data_dev_idx + data_test_idx
(data_train_idx, data_dev_idx, data_test_500_seen_idx, data_test_500_unseen_idx, data_defi_c_idx, data_desc_c_idx) = np.load(BASE_DIR + 'data_inUse2_en.npy', allow_pickle=True)
data_all_idx = data_train_idx + data_dev_idx + data_test_500_seen_idx + data_test_500_unseen_idx + data_defi_c_idx
index2word_en = np.array(index2word_en)
print('-------------------------2')
sememe_num = len(index2sememe)
wd2sem = word2feature(data_all_idx, label_size, sememe_num, 'sememes')
wd_sems_ = label_multihot(wd2sem, sememe_num)
wd_sems_ = torch.from_numpy(np.array(wd_sems_)).to(device) 
lexname_num = len(index2lexname)
wd2lex = word2feature(data_all_idx, label_size, lexname_num, 'lexnames') 
wd_lex = label_multihot(wd2lex, lexname_num)
wd_lex = torch.from_numpy(np.array(wd_lex)).to(device)
rootaffix_num = len(index2rootaffix)
wd2ra = word2feature(data_all_idx, label_size, rootaffix_num, 'root_affix') 
wd_ra = label_multihot(wd2ra, rootaffix_num)
wd_ra = torch.from_numpy(np.array(wd_ra)).to(device)
mask_s_ = mask_noFeature(label_size, wd2sem, sememe_num)
mask_l = mask_noFeature(label_size, wd2lex, lexname_num)
mask_r = mask_noFeature(label_size, wd2ra, rootaffix_num)
#del data_all_idx, data_train_idx, data_dev_idx, data_test_idx
del data_all_idx, data_train_idx, data_dev_idx, data_test_500_seen_idx, data_test_500_unseen_idx, data_defi_c_idx
gc.collect()
print('-------------------------3')
# 添加wordnet synset用于描述为一个词时的同义词推荐
index2synset_en = [[] for i in range(len(word2index_en))]
for line in open(BASE_DIR + 'word_synsetWords.txt').readlines():
    wd = line.split()[0]
    synset = line.split()[1:]
    for syn in synset:
        index2synset_en[word2index_en[wd]].append(word2index_en[syn])
print('-------------------------4')
model_en = torch.load(MODEL_FILE_en, map_location=lambda storage, loc: storage)
model_en.eval()
print('-------------------------5')
def home(request):
    return render(request, 'home.html')
    
def admin(request):
    result = json.load(open('datastatistics.current', 'r'))
    [updatetime, pageview, totalqueries, uniquevisitor, effectiveflow, weeknum, weekvalue, month2019v, month2020v, visit2019v, visit2020v, feedbackinfo] = result
    
    pageview = format(pageview, ',')
    totalqueries = format(totalqueries, ',')
    uniquevisitor = format(uniquevisitor, ',')
    effectiveflow = format(effectiveflow, ',')
    def fixvalue2str(value):
        i = -1
        while(True): # fix the value if 0
            if value[i] == 0:
                value[i] = -1
                i -= 1
            else:
                break
        value = [i+1 for i in value]
        value = str(value).replace(', 0', ', ') # replace 0 to null for painting
        return value
    weekvalue = fixvalue2str(weekvalue)
    #month2019v = fixvalue2str(month2019v)
    #month2020v = fixvalue2str(month2020v)

    return render(request, 'admin.html', context=locals())
    #return render(request, 'admin.html')

def about(request):
    return render(request, 'about.html')
    
def about_en(request):
    return render(request, 'about_en.html')
    
def papers(request):
    return render(request, 'papers.html')
    
def help(request):
    return render(request, 'help.html')
    
def Score2Hexstr(score, maxsc):
    thr = maxsc/1.5
    l = len(score)
    ret = ['00']*l
    for i in range(l):
        res = int(200*(score[i] - thr)/thr)
        if res>15:
            ret[i] = hex(res)[2:]
        else:
            break
    return ret
    
def ChineseRD(request):
    description = request.GET['description']
    RD_mode = request.GET['mode']
    if RD_mode=='EC':
        q = description
        fromLang = 'en'
        toLang = 'zh'
        salt = "35555"
        sign = appid+q+salt+secretKey
        sign = md5(sign)
        url = "http://api.fanyi.baidu.com/api/trans/vip/translate"
        url = url + '?appid='+appid+'&q='+urllib.parse.quote(q)+'&from='+fromLang+'&to='+toLang+'&salt='+str(salt)+'&sign='+sign
        response = requests.request("GET", url)
        description = eval(response.text)['trans_result'][0]['dst']
    with torch.no_grad():
        if description == "你好":
            description = "你好？"
        def_words = [w for w, p in lac.cut(description)]
        def_word_idx = []
        if len(def_words) > 0:
            for def_word in def_words:
                if def_word in word2index:
                    def_word_idx.append(word2index[def_word])
                else:
                    #======================================= word cut to char when not in word2vec
                    for dw in def_word:
                        try:
                            def_word_idx.append(word2index[dw])
                        except:
                            def_word_idx.append(word2index['<OOV>'])
                    #=======================================
            x_len = len(def_word_idx)
            if set(def_word_idx)=={word2index['<OOV>']}:
                x_len = 1
            if x_len==1:
                if def_word_idx[0]>1:
                    #词向量找相关词，排序后，如果在词林里，则对应的同义词的分数乘以2？
                    score = ((model.embedding.weight.data).mm((model.embedding.weight.data[def_word_idx[0]]).unsqueeze(1))).squeeze(1)
                    if RD_mode=='CC': #当CC的时候，排除自身，EC的时候自身是最准确的，不排除。
                        score[def_word_idx[0]] = -10.
                    score[np.array(index2synset[def_word_idx[0]])] *= 2
                    sc, indices = torch.sort(score, descending=True)
                    predicted = indices[:NUM_RESPONSE].detach().cpu().numpy()
                    score = sc[:NUM_RESPONSE].detach().numpy()
                    maxsc = sc[0].detach().item()
                    s2h = Score2Hexstr(score, maxsc)
                else:
                    predicted= []
                    ret = {'error': 1} # 字符无法识别
            else:
                defi = '[CLS] ' + description
                def_word_idx = tokenizer_Ch.encode(defi)[:80]
                def_word_idx.extend(tokenizer_Ch.encode('[SEP]'))
                definition_words_t = torch.tensor(np.array(def_word_idx), dtype=torch.int64, device=device)
                definition_words_t = definition_words_t.unsqueeze(0) # batch_size = 1
                score = model('test', x=definition_words_t, w=words_t, ws=wd_sems, wP=wd_POSs, wc=wd_charas, wC=wd_C, msk_s=mask_s, msk_c=mask_c, mode=MODE)
                sc, indices = torch.sort(score, descending=True)
                predicted = indices[0, :NUM_RESPONSE].detach().cpu().numpy()
                score = sc[0, :NUM_RESPONSE].detach().numpy()
                maxsc = sc[0, 0].detach().item()
                s2h = Score2Hexstr(score, maxsc)
        else:
            predicted= []
            ret = {'error': 0} # 输入为空
    if len(predicted)>0:
        res = index2word[predicted]
        ret = [] # 不能以字典形式返回，因为字典是无序的，这里需要用列表来保持顺序。
        cn = -1
        if RD_mode=='CC':
            def_words = set(def_words)
            for wd in res:
                cn += 1
                if wd not in def_words:
                    try:
                        ret.append(wd_data[wd]) # wd_data[wd] = {'word':字典, 'definition':defis, 'POS':['名'], 'bihuashu':14, 'bihuashu1st':6, 'length':2, 'pinyin':'zì diǎn', 'pinyinshouzimu': 'z d'}]
                        ret[len(ret)-1]['c'] = s2h[cn]
                    except:
                        continue
        else:
            for wd in res:
                cn += 1
                try:
                    ret.append(wd_data[wd]) # wd_data[wd] = {'word':字典, 'definition':defis, 'POS':['名'], 'bihuashu':14, 'bihuashu1st':6, 'length':2, 'pinyin':'zì diǎn', 'pinyinshouzimu': 'z d'}]
                    ret[len(ret)-1]['c'] = s2h[cn]
                except:
                    continue
    return HttpResponse(json.dumps(ret,ensure_ascii=False),content_type="application/json,charset=utf-8")

def getClass2Class(r, score): # 聚类时的排序是随机的，这里根据每类的前5个计算各类的平均分，按分高低排序各类顺序（按新顺序给类编号）
    perCluster = [[],[],[],[],[],[]]
    for i in range(GET_NUM):
        perCluster[r[i]].append(score[i])
    scorePC = []
    for i in range(6):
        l = len(perCluster[i]) if len(perCluster[i])<5 else 5
        scorePC.append(sum(perCluster[i][:l])/l)
    ind = [indsc[0] for indsc in sorted(enumerate(scorePC), key=lambda x:x[1], reverse=True)]
    class2class = [0,0,0,0,0,0]
    for i in range(6):
        class2class[ind[i]] = i
    return class2class
                    
def ChineseRDCluster(request):
    description = request.GET['description']
    RD_mode = request.GET['mode']
    if RD_mode=='EC':
        q = description
        fromLang = 'en'
        toLang = 'zh'
        salt = "35555"
        sign = appid+q+salt+secretKey
        sign = md5(sign)
        url = "http://api.fanyi.baidu.com/api/trans/vip/translate"
        url = url + '?appid='+appid+'&q='+urllib.parse.quote(q)+'&from='+fromLang+'&to='+toLang+'&salt='+str(salt)+'&sign='+sign
        response = requests.request("GET", url)
        description = eval(response.text)['trans_result'][0]['dst']
    with torch.no_grad():
        if description == "你好":
            description = "你好？"
        def_words = [w for w, p in lac.cut(description)]
        def_word_idx = []
        if len(def_words) > 0:
            for def_word in def_words:
                if def_word in word2index:
                    def_word_idx.append(word2index[def_word])
                else:
                    #======================================= word cut to char when not in word2vec
                    for dw in def_word:
                        try:
                            def_word_idx.append(word2index[dw])
                        except:
                            def_word_idx.append(word2index['<OOV>'])
                    #=======================================
            x_len = len(def_word_idx)
            if set(def_word_idx)=={word2index['<OOV>']}:
                x_len = 1
            if x_len==1:
                if def_word_idx[0]>1:
                    #词向量找相关词，排序后，如果在词林里，则对应的同义词的分数乘以2？
                    score = ((model.embedding.weight.data).mm((model.embedding.weight.data[def_word_idx[0]]).unsqueeze(1))).squeeze(1)
                    if RD_mode=='CC': #当CC的时候，排除自身，EC的时候自身是最准确的，不排除。
                        score[def_word_idx[0]] = -10.
                    score[np.array(index2synset[def_word_idx[0]])] *= 2
                    sc, indices = torch.sort(score, descending=True)
                    predicted = indices[:GET_NUM].detach().cpu().numpy()
                    score = sc[:GET_NUM].detach().numpy()
                    maxsc = sc[0].detach().item()
                    s2h = Score2Hexstr(score, maxsc)
                    r = kmeans.fit_predict(model.embedding.weight.data[predicted[:GET_NUM]].cpu().numpy()) # GET_NUM
                    class2class = getClass2Class(r, score[:GET_NUM])
                else:
                    predicted= []
                    ret = {'error': 1} # 字符无法识别
            else:
                defi = '[CLS] ' + description
                def_word_idx = tokenizer_Ch.encode(defi)[:80]
                def_word_idx.extend(tokenizer_Ch.encode('[SEP]'))
                definition_words_t = torch.tensor(np.array(def_word_idx), dtype=torch.int64, device=device)
                definition_words_t = definition_words_t.unsqueeze(0) # batch_size = 1
                score = model('test', x=definition_words_t, w=words_t, ws=wd_sems, wP=wd_POSs, wc=wd_charas, wC=wd_C, msk_s=mask_s, msk_c=mask_c, mode=MODE)
                sc, indices = torch.sort(score, descending=True)
                predicted = indices[0, :GET_NUM].detach().cpu().numpy()
                score = sc[0, :GET_NUM].detach().numpy()
                maxsc = sc[0, 0].detach().item()
                s2h = Score2Hexstr(score, maxsc)
                r = kmeans.fit_predict(model.embedding.weight.data[predicted[:GET_NUM]].cpu().numpy()) # GET_NUM
                class2class = getClass2Class(r, score[:GET_NUM])
        else:
            predicted= []
            ret = {'error': 0} # 输入为空
    if len(predicted)>0:
        res = index2word[predicted]
        ret = [] # 不能以字典形式返回，因为字典是无序的，这里需要用列表来保持顺序。
        cn = -1
        if RD_mode=='CC':
            def_words = set(def_words)
            for wd in res:
                cn += 1
                if wd not in def_words:
                    try:
                        ret.append(wd_data[wd]) # wd_data[wd] = {'word':字典, 'definition':defis, 'POS':['名'], 'bihuashu':14, 'bihuashu1st':6, 'length':2, 'pinyin':'zì diǎn', 'pinyinshouzimu': 'z d'}]
                        ret[len(ret)-1]['c'] = s2h[cn]
                        ret[len(ret)-1]['C'] = class2class[int(r[cn])] # 必须转为int，否则其实是int64类型，会报不能json序列化的错误
                        ret[len(ret)-1]['d'] = wd_defi[wd]
                        ret.sort(key=lambda x: x['C'])
                    except:
                        continue
        else:
            for wd in res:
                cn += 1
                try:
                    ret.append(wd_data[wd]) # wd_data[wd] = {'word':字典, 'definition':defis, 'POS':['名'], 'bihuashu':14, 'bihuashu1st':6, 'length':2, 'pinyin':'zì diǎn', 'pinyinshouzimu': 'z d'}]
                    ret[len(ret)-1]['c'] = s2h[cn]
                    ret[len(ret)-1]['C'] = class2class[int(r[cn])] # 必须转为int，否则其实是int64类型，会报不能json序列化的错误
                    ret[len(ret)-1]['d'] = wd_defi[wd]
                    ret.sort(key=lambda x: x['C'])
                except:
                    continue
    return HttpResponse(json.dumps(ret,ensure_ascii=False),content_type="application/json,charset=utf-8")
      
def EnglishRDCluster(request):
    description = request.GET['description']
    RD_mode = request.GET['mode']
    if RD_mode=='CE':
        filter = re.compile(r"[\u4e00-\u9fa5]+")
        desc = ''.join(filter.findall(description))
        def_words = [w for w, p in lac.cut(desc)]
        q = description
        fromLang = 'zh'
        toLang = 'en'
        salt = "35555"
        sign = appid+q+salt+secretKey
        sign = md5(sign)
        url = "http://api.fanyi.baidu.com/api/trans/vip/translate"
        url = url + '?appid='+appid+'&q='+urllib.parse.quote(q)+'&from='+fromLang+'&to='+toLang+'&salt='+str(salt)+'&sign='+sign
        response = requests.request("GET", url)
        description = eval(response.text)['trans_result'][0]['dst']
    with torch.no_grad():
        def_words = re.sub('[%s]' % re.escape(string.punctuation), ' ', description)
        def_words = def_words.lower()
        def_words = def_words.strip().split()
        def_word_idx = []
        if len(def_words) > 0:
            for def_word in def_words:
                if def_word in word2index_en:
                    def_word_idx.append(word2index_en[def_word])
                else:
                    def_word_idx.append(word2index_en['<OOV>'])
            x_len = len(def_word_idx)
            if set(def_word_idx)=={word2index_en['<OOV>']}:
                x_len = 1
            if x_len==1:
                if def_word_idx[0]>1:
                    #词向量找相关词，排序后，如果在WordNet的synset里，则对应的同义词的分数乘以2？
                    score = ((model_en.embedding.weight.data).mm((model_en.embedding.weight.data[def_word_idx[0]]).unsqueeze(1))).squeeze(1)
                    if RD_mode=='EE': #当EE的时候，排除自身，CE的时候自身是最准确的，不排除。
                        score[def_word_idx[0]] = -10.
                    score[np.array(index2synset_en[def_word_idx[0]])] *= 2
                    sc, indices = torch.sort(score, descending=True)
                    predicted = indices[:GET_NUM].detach().cpu().numpy()
                    
                    score = sc[:GET_NUM].detach().numpy()
                    maxsc = sc[0].detach().item()
                    s2h = Score2Hexstr(score, maxsc)
                    r = kmeans.fit_predict(model.embedding.weight.data[predicted[:GET_NUM]].cpu().numpy()) # GET_NUM
                    class2class = getClass2Class(r, score[:GET_NUM])
                else:
                    predicted= []
                    ret = {'error': 1} # 字符无法识别
            else:
                defi = '[CLS] ' + description
                def_word_idx = tokenizer_En.encode(defi)[:60]
                def_word_idx.extend(tokenizer_En.encode('[SEP]'))
                definition_words_t = torch.tensor(np.array(def_word_idx), dtype=torch.int64, device=device)
                definition_words_t = definition_words_t.unsqueeze(0) # batch_size = 1
                score = model_en('test', x=definition_words_t, w=words_t, ws=wd_sems_, wl=wd_lex, wr=wd_ra, msk_s=mask_s_, msk_l=mask_l, msk_r=mask_r, mode=MODE_en)
                sc, indices = torch.sort(score, descending=True)
                predicted = indices[0, :GET_NUM].detach().cpu().numpy()
                score = sc[0, :GET_NUM].detach().numpy()
                maxsc = sc[0, 0].detach().item()
                s2h = Score2Hexstr(score, maxsc)
                r = kmeans.fit_predict(model.embedding.weight.data[predicted[:GET_NUM]].cpu().numpy()) # GET_NUM
                class2class = getClass2Class(r, score[:GET_NUM])
        else:
            predicted= []
            ret = {'error': 0} # 输入为空
    if len(predicted)>0:
        res = index2word_en[predicted]
        ret = [] # 不能以字典形式返回，因为字典是无序的，这里需要用列表来保持顺序。
        cn = -1
        if RD_mode == "EE":
            def_words = set(def_words)
            for wd in res:
                cn += 1
                if len(wd)>1 and (wd not in def_words):
                    try:
                        ret.append(wd_data_en[wd]) # wd_data_en[wd] = {'word': word, 'definition':defis, 'POS':['n']}]
                        ret[len(ret)-1]['c'] = s2h[cn]
                        ret[len(ret)-1]['C'] = class2class[int(r[cn])] # 必须转为int，否则其实是int64类型，会报不能json序列化的错误
                        ret[len(ret)-1]['d'] = wd_defi_en[wd]
                        ret.sort(key=lambda x: x['C'])
                    except:
                        continue
        else:
            for wd in res:
                cn += 1
                if len(wd)>1:
                    try:
                        ret.append(wd_data_en[wd]) # wd_data_en[wd] = {'word': word, 'definition':defis, 'POS':['n']}]
                        ret[len(ret)-1]['c'] = s2h[cn]
                        ret[len(ret)-1]['C'] = class2class[int(r[cn])] # 必须转为int，否则其实是int64类型，会报不能json序列化的错误
                        ret[len(ret)-1]['d'] = wd_defi_en[wd]
                        ret.sort(key=lambda x: x['C'])
                    except:
                        continue
    return HttpResponse(json.dumps(ret,ensure_ascii=False),content_type="application/json,charset=utf-8")
    

def EnglishRD(request):
    description = request.GET['description']
    RD_mode = request.GET['mode']
    if RD_mode=='CE':
        q = description
        fromLang = 'zh'
        toLang = 'en'
        salt = "35555"
        sign = appid+q+salt+secretKey
        sign = md5(sign)
        url = "http://api.fanyi.baidu.com/api/trans/vip/translate"
        url = url + '?appid='+appid+'&q='+urllib.parse.quote(q)+'&from='+fromLang+'&to='+toLang+'&salt='+str(salt)+'&sign='+sign
        response = requests.request("GET", url)
        description = eval(response.text)['trans_result'][0]['dst']
        #print(description)
    with torch.no_grad():
        def_words = re.sub('[%s]' % re.escape(string.punctuation), ' ', description)
        def_words = def_words.lower()
        def_words = def_words.strip().split()
        def_word_idx = []
        if len(def_words) > 0:
            for def_word in def_words:
                if def_word in word2index_en:
                    def_word_idx.append(word2index_en[def_word])
                else:
                    def_word_idx.append(word2index_en['<OOV>'])
            x_len = len(def_word_idx)
            if set(def_word_idx)=={word2index_en['<OOV>']}:
                x_len = 1
            if x_len==1:
                if def_word_idx[0]>1:
                    #词向量找相关词，排序后，如果在WordNet的synset里，则对应的同义词的分数乘以2？
                    score = ((model_en.embedding.weight.data).mm((model_en.embedding.weight.data[def_word_idx[0]]).unsqueeze(1))).squeeze(1)
                    if RD_mode=='EE': #当EE的时候，排除自身，CE的时候自身是最准确的，不排除。
                        score[def_word_idx[0]] = -10.
                    score[np.array(index2synset_en[def_word_idx[0]])] *= 2
                    sc, indices = torch.sort(score, descending=True)
                    predicted = indices[:NUM_RESPONSE].detach().cpu().numpy()
                    
                    score = sc[:NUM_RESPONSE].detach().numpy()
                    maxsc = sc[0].detach().item()
                    s2h = Score2Hexstr(score, maxsc)
                else:
                    predicted= []
                    ret = {'error': 1} # 字符无法识别
            else:
                defi = '[CLS] ' + description
                def_word_idx = tokenizer_En.encode(defi)[:60]
                def_word_idx.extend(tokenizer_En.encode('[SEP]'))
                definition_words_t = torch.tensor(np.array(def_word_idx), dtype=torch.int64, device=device)
                definition_words_t = definition_words_t.unsqueeze(0) # batch_size = 1
                score = model_en('test', x=definition_words_t, w=words_t, ws=wd_sems_, wl=wd_lex, wr=wd_ra, msk_s=mask_s_, msk_l=mask_l, msk_r=mask_r, mode=MODE_en)
                sc, indices = torch.sort(score, descending=True)
                predicted = indices[0, :NUM_RESPONSE].detach().cpu().numpy()
                score = sc[0, :NUM_RESPONSE].detach().numpy()
                maxsc = sc[0, 0].detach().item()
                s2h = Score2Hexstr(score, maxsc)
                
        else:
            predicted= []
            ret = {'error': 0} # 输入为空
    if len(predicted)>0:
        res = index2word_en[predicted]
        ret = [] # 不能以字典形式返回，因为字典是无序的，这里需要用列表来保持顺序。
        cn = -1
        if RD_mode == "EE":
            def_words = set(def_words)
            for wd in res:
                cn += 1
                if len(wd)>1 and (wd not in def_words):
                    try:
                        ret.append(wd_data_en[wd]) # wd_data_en[wd] = {'word': word, 'definition':defis, 'POS':['n']}]
                        ret[len(ret)-1]['c'] = s2h[cn]
                    except:
                        continue
        else:
            for wd in res:
                cn += 1
                if len(wd)>1:
                    try:
                        ret.append(wd_data_en[wd]) # wd_data_en[wd] = {'word': word, 'definition':defis, 'POS':['n']}]
                        ret[len(ret)-1]['c'] = s2h[cn]
                    except:
                        continue
    return HttpResponse(json.dumps(ret,ensure_ascii=False),content_type="application/json,charset=utf-8")
    
    
def feedback(request):
    content = request.GET['content']
    FBmode = request.GET['mode']
    if FBmode=='FBS':
        f = open('./feedBackLog/'+datetime.now().date().strftime('%Y%m')+'suggestion.log', 'a')
        f.write(datetime.now().strftime('[%Y%m%d%H%M%S] ')+content+'\n')
    elif FBmode=='FBW':
        f = open('./feedBackLog/'+datetime.now().date().strftime('%Y%m')+'wordsDesc.log', 'a')
        f.write(datetime.now().strftime('[%Y%m%d%H%M%S] ')+content+'\n')
    f.close()
    return HttpResponse("")

def GetChDefis(request):
    if(request.method == 'POST'):
        words = request.POST['w'].split()
    else: # GET method
        words = request.GET['w'].split()
    ret = []
    for w in words:
        ret.append(wd_defi[w])
    return HttpResponse(json.dumps(ret,ensure_ascii=False),content_type="application/json,charset=utf-8")
    
def GetEnDefis(request):
    if(request.method == 'POST'):
        words = request.POST['w'].split()
    else: # GET method
        words = request.GET['w'].split()
    ret = []
    for w in words:
        ret.append(wd_defi_en[w])
    return HttpResponse(json.dumps(ret,ensure_ascii=False),content_type="application/json,charset=utf-8")
