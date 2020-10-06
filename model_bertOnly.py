import torch

class Encoder(torch.nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, layers, class_num, encoder):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = 200 # english: 300
        self.hidden_dim = 768
        self.layers = layers
        self.class_num = class_num
        self.embedding = torch.nn.Embedding(self.vocab_size, self.embed_dim, padding_idx=0, max_norm=5, sparse=True)
        self.embedding.weight.requires_grad = False
        self.embedding_dropout = torch.nn.Dropout(0.2)
        self.encoder = encoder
        self.fc = torch.nn.Linear(self.hidden_dim, self.embed_dim)
        self.loss = torch.nn.CrossEntropyLoss()

        
    def forward(self, operation, x=None, w=None):
        # x: T(bat, max_word_num)
        # w: T(bat)
        # h: T(bat, max_word_num, 768)
        attention_mask = torch.gt(x, 0).to(torch.int64)
        h = self.encoder(x, attention_mask=attention_mask)[0]
        h_1 = self.embedding_dropout(h[:,0,:])
        vd = self.fc(h_1)
        # score0: T(bat, 30000) = [bat, emb] .mm [class_num, emb].t()
        score = vd.mm(self.embedding.weight.data[[range(self.class_num)]].t())
        # fine-tune depended on the target word shouldn't exist in the definition.
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
            return indices
