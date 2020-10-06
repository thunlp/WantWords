import torch
import numpy as np

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
    def __init__(self, vocab_size, embed_dim, hidden_dim, layers, class_num, sememe_num, lexname_num, rootaffix_num):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.class_num = class_num
        self.sememe_num = sememe_num
        self.lexname_num = lexname_num
        self.rootaffix_num = rootaffix_num
        self.embedding = torch.nn.Embedding(self.vocab_size, self.embed_dim, padding_idx=0, max_norm=5, sparse=True)
        self.embedding.weight.requires_grad = False
        self.embedding_dropout = torch.nn.Dropout()
        self.encoder = BiLSTM(self.embed_dim, self.hidden_dim, self.layers)
        self.fc = torch.nn.Linear(self.hidden_dim*2, self.embed_dim)
        self.fc_s = torch.nn.Linear(self.hidden_dim*2, self.sememe_num)
        self.fc_l = torch.nn.Linear(self.hidden_dim*2, self.lexname_num)
        self.fc_r = torch.nn.Linear(self.hidden_dim*2, self.rootaffix_num)
        self.loss = torch.nn.CrossEntropyLoss()
        self.relu = torch.nn.ReLU()

        
    def forward(self, operation, x=None, w=None, ws=None, wl=None, wr=None, msk_s=None, msk_l=None, msk_r=None, mode=None, RD_mode=None):
        # x: T(bat, max_word_num)
        # w: T(bat)
        # x_embedding: T(bat, max_word_num, embed_dim)
        x_embedding = self.embedding(x)
        x_embedding = self.embedding_dropout(x_embedding)
        # mask: T(bat, max_word_num)
        mask = torch.gt(x, 0).to(torch.int64)
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
        h_1 = torch.sum(h*alpha, 1)
        vd = self.fc(h_1) #+ torch.sum(self.embedding(x), 1)#+ torch.sum(x_embedding, 1) #ok
        #vd = self.fc(torch.sum(torch.cat([h, self.embedding(x)], 2)*alpha, 1)) #best
        score0 = vd.mm(self.embedding.weight[[range(self.class_num)]].t())
        score = score0
        if 's' in mode:
            ## sememe prediction
            # pos_score: T(bat, max_word_num, sememe_num)
            pos_score = self.fc_s(h)
            pos_score = pos_score*mask_3 + (-1e7)*(1-mask_3)
            # sem_score: T(bat, sememe_num)
            sem_score, _ = torch.max(pos_score, dim=1)
            #sem_score = torch.sum(pos_score * alpha, 1)
            # score: T(bat, class_num) = [bat, sememe_num] .mm [class_num, sememe_num].t()
            score_s = self.relu(sem_score.mm(ws.t()))
            #----------add mean sememe score to those who have no sememes
            # mean_sem_sc: T(bat)
            mean_sem_sc = torch.mean(score_s, 1)
            # msk: T(class_num)
            score_s = score_s + mean_sem_sc.unsqueeze(1).mm(msk_s.unsqueeze(0))
            #----------
            score = score + score_s
        if 'r' in mode:
            ## root-affix prediction
            pos_score_ = self.fc_r(h)
            pos_score_ = pos_score_*mask_3 + (-1e7)*(1-mask_3)
            ra_score, _ = torch.max(pos_score_, dim=1)
            score_r = self.relu(ra_score.mm(wr.t()))
            mean_ra_sc = torch.mean(score_r, 1)
            score_r = score_r + mean_ra_sc.unsqueeze(1).mm(msk_r.unsqueeze(0))
            score = score + score_r
        if 'l' in mode:
            ## lexname prediction
            lex_score = self.fc_l(h_1)
            score_l = self.relu(lex_score.mm(wl.t()))
            mean_lex_sc = torch.mean(score_l, 1)
            score_l = score_l + mean_lex_sc.unsqueeze(1).mm(msk_l.unsqueeze(0))
            '''
            pos_score__ = self.fc_l(h)
            pos_score__ = pos_score__*mask_3 + (-1e7)*(1-mask_3)
            l_score, _ = torch.max(pos_score__, dim=1)
            score_l = self.relu(l_score.mm(wl.t()))
            mean_l_sc = torch.mean(score_l, 1)
            score_l = score_l + mean_l_sc.unsqueeze(1).mm(msk_l.unsqueeze(0))
            '''
            score = score + score_l
        
        if RD_mode != 'CE': # EE or EO
            # fine-tune depended on the target word shouldn't exist in the definition.
            #score_res = score.clone().detach()
            mask1 = torch.lt(x, self.class_num).to(torch.int64)
            mask2 = torch.ones((score.shape[0], score.shape[1]), dtype=torch.float32)
            for i in range(x.shape[0]):
                mask2[i][x[i]*mask1[i]] = 0.
            score = score * mask2 + (-1e6)*(1-mask2)
        
        _, indices = torch.sort(score, descending=True)
        if operation == 'train':
            loss = self.loss(score, w)
            return loss, score, indices
        elif operation == 'test':
            return score, indices

