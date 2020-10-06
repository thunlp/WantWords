import torch

class BiLSTM(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
    
    def forward(self, x, x_len):
        # x: T(bat, len, emb) float32
        # x_len: T(bat) int64
        _, x_len_sort_idx = torch.sort(-x_len)
        _, x_len_unsort_idx = torch.sort(x_len_sort_idx)
        x = x[x_len_sort_idx]
        x_len = x_len[x_len_sort_idx]
        x_packed = torch.nn.utils.rnn.pack_padded_sequence(x, x_len, batch_first=True)
        # ht: T(num_layers*2, bat, hid) float32
        # ct: T(num_layers*2, bat, hid) float32
        h_packed, (ht, ct) = self.lstm(x_packed, None)
        ht = ht[:, x_len_unsort_idx, :]
        ct = ct[:, x_len_unsort_idx, :]
        # h: T(bat, len, hid*2) float32
        h, _ = torch.nn.utils.rnn.pad_packed_sequence(h_packed, batch_first=True)
        h = h[x_len_unsort_idx]
        return h, (ht, ct)
        
class Encoder(torch.nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, layers, class_num, sememe_num, chara_num, mode):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.class_num = class_num
        self.sememe_num = sememe_num
        self.chara_num = chara_num
        self.embedding = torch.nn.Embedding(self.vocab_size, self.embed_dim, padding_idx=0, max_norm=5, sparse=True)
        self.embedding.weight.requires_grad = False
        self.embedding_dropout = torch.nn.Dropout()
        self.encoder = BiLSTM(self.embed_dim, self.hidden_dim, self.layers)
        self.fc = torch.nn.Linear(self.hidden_dim*2, self.embed_dim)
        #self.fc = torch.nn.Linear(self.hidden_dim*2+self.embed_dim, self.embed_dim)
        self.loss = torch.nn.CrossEntropyLoss()
        self.relu = torch.nn.ReLU()
        if 'P' in mode:
            self.fc2 = torch.nn.Linear(self.hidden_dim*2, 13)
        if 's' in mode:
            self.fc1 = torch.nn.Linear(self.hidden_dim*2, self.sememe_num)
        if 'c' in mode:
            self.fc3 = torch.nn.Linear(self.hidden_dim*2, self.chara_num)
        if 'C' in mode:
            self.fc_C1 = torch.nn.Linear(self.hidden_dim*2, 12)
            self.fc_C2 = torch.nn.Linear(self.hidden_dim*2, 95)
            self.fc_C3 = torch.nn.Linear(self.hidden_dim*2, 1425)
        
    def forward(self, operation, x=None, w=None, ws=None, wP=None, wc=None, wC=None, msk_s=None, msk_c=None, mode=None, RD_mode=None):
        # x: T(bat, max_word_num)
        # w: T(bat)
        # x_embedding: T(bat, max_word_num, embed_dim)
        x = x.long()
        x_embedding = self.embedding(x)
        x_embedding = self.embedding_dropout(x_embedding)
        # mask: T(bat, max_word_num)
        mask = torch.gt(x, 0).to(torch.int32)
        # x_len: T(bat)
        x_len = torch.sum(mask, dim=1)
        # h: T(bat, max_word_num, hid*2)
        # ht: T(num_layers*2, bat, hid) float32
        h, (ht, _) = self.encoder(x_embedding, x_len)
        # ht: T(bat, hid*2)
        ht = torch.transpose(ht[ht.shape[0] - 2:, :, :], 0, 1).contiguous().view(x_len.shape[0], self.hidden_dim*2)
        # alpha: T(bat, max_word_num, 1)
        alpha = (h.bmm(ht.unsqueeze(2)))
        # mask_3: T(bat, max_word_num, 1)
        mask_3 = mask.to(torch.float32).unsqueeze(2)

        ## word prediction
        # vd: T(bat, embed_dim)
        #h_ = self.fc(h)
        #vd = torch.sum(h_, 1)
        #vd = self.fc(torch.max(h, dim=1)[0])
        h_1 = torch.sum(h*alpha, 1)
        vd = self.fc(h_1) #+ torch.sum(self.embedding(x), 1)#+ torch.sum(x_embedding, 1) #ok
        #vd = self.fc(torch.cat([torch.sum(h*alpha, 1), torch.sum(self.embedding(x), 1)], 1)) #ok
        #vd = self.fc(torch.sum(torch.cat([h, self.embedding(x)], 2)*alpha, 1)) #best
        #vd = self.fc(torch.max(torch.cat([h, self.embedding(x)], 2), 1)[0]) #bad
        #vd = torch.sum(self.embedding(x), 1)
        #vd = torch.max(self.embedding.weight(x), 1)[0]
        #vd = self.fc(ht)
        # score: T(bat, calss_num)
        score0 = vd.mm(self.embedding.weight[[range(self.class_num)]].t())
        score = score0
        
        if 'C' in mode:
            # scC[i]: T(bat, Ci_size)
            # 词林的层次分类训练的慢，其实这样不公平，不平衡，因为词预测先收敛了，而cilin的分类还没效果，其他信息的利用也有同样的问题，不一定同时收敛！！！
            scC = [self.fc_C1(h_1), self.fc_C2(h_1), self.fc_C3(h_1)]
            score2 = torch.zeros((score0.shape[0], score0.shape[1]), dtype=torch.float32)
            rank = 0.6
            for i in range(3):
                # wC[i]: T(class_num, Ci_size)
                # C_sc: T(bat, class_num)
                score2 += self.relu(scC[i].mm(wC[i].t())*(rank**i))
            #----------add mean cilin-class score to those who have no cilin-class
            mean_cilin_sc = torch.mean(score2, 1)
            score2 = score2*(1-msk_c) + mean_cilin_sc.unsqueeze(1).mm(msk_c.unsqueeze(0))
            #----------
            score = score + score2/2
        if 'P' in mode:
            ## POS prediction
            # score_POS: T(bat, 13) pos_num=12+1
            score_POS = self.fc2(torch.sum(h*alpha, 1))
            # s: (class_num, 13) multi-hot
            # weight_sc: T(bat, class_num) = [bat, 13] .mm [class_num, 13].t()
            weight_sc = self.relu(score_POS.mm(wP.t()))
            #print(torch.max(weight_sc), torch.min(weight_sc))
            score = score + weight_sc
        if 's' in mode:
            ## sememe prediction
            # pos_score: T(bat, max_word_num, sememe_num)
            pos_score = self.fc1(h)
            pos_score = pos_score*mask_3 + (-1e7)*(1-mask_3)
            # sem_score: T(bat, sememe_num)
            sem_score, _ = torch.max(pos_score, dim=1)
            #sem_score = torch.sum(pos_score * alpha, 1)
            # score: T(bat, class_num) = [bat, sememe_num] .mm [class_num, sememe_num].t()
            score1 = self.relu(sem_score.mm(ws.t()))
            #----------add mean sememe score to those who have no sememes
            # mean_sem_sc: T(bat)
            mean_sem_sc = torch.mean(score1, 1)
            # msk: T(class_num)
            score1 = score1 + mean_sem_sc.unsqueeze(1).mm(msk_s.unsqueeze(0))
            #----------
            score = score + score1
        if 'c' in mode:
            ## character prediction
            # pos_score: T(bat, max_word_num, sememe_num)
            pos_score = self.fc3(h)
            pos_score = pos_score*mask_3 + (-1e7)*(1-mask_3)
            # chara_score: T(bat, chara_num)
            chara_score, _ = torch.max(pos_score, dim=1)
            #chara_score = torch.sum(pos_score * alpha, 1)
            # score: T(bat, class_num) = [bat, sememe_num] .mm [class_num, sememe_num].t()
            score3 = self.relu(chara_score.mm(wc.t()))
            score = score + score3
        
        if RD_mode == 'CC':
            # fine-tune depended on the target word shouldn't exist in the definition.
            #score_res = score.clone().detach()
            mask1 = torch.lt(x, self.class_num).to(torch.int64)
            mask2 = torch.ones((score.shape[0], score.shape[1]), dtype=torch.float32)
            for i in range(x.shape[0]):
                mask2[i][x[i]*mask1[i]] = 0.
            score = score * mask2 + (-1e6)*(1-mask2)
        
        _, indices = torch.sort(score, descending=True)
        if operation == 'train':
            loss = self.loss(score, w.long())
            return loss, score, indices
        elif operation == 'test':
            return score, indices
        elif operation == 'getSentenceEncoding':
            # vd: T(bat, embed_dim)
            return vd, score, indices
