import torch.nn as nn
import torch.nn.functional as F

class RNNClassifier(nn.Module):

    def __init__(
        self,
        input_size,
        word_vec_size,
        hidden_size,
        n_classes,
        n_layers=4,
        dropout_p=0.3,
        pretrained_embedding=None,
        freeze_embedding=False,
    ):
        self.input_size = input_size  # vocabulary_size
        self.word_vec_size = word_vec_size
        self.hidden_size = hidden_size
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.dropout_p = dropout_p

        super().__init__()

        if pretrained_embedding is not None:
            print("doing with pretrained model!!!")
            self.input_size, self.word_vec_size = pretrained_embedding.shape
            self.emb = nn.Embedding.from_pretrained(pretrained_embedding,
                                                    freeze=freeze_embedding)
        else:
            print("doing without pretrained model!!!")
            self.word_vec_size = word_vec_size
            self.emb = nn.Embedding(input_size, word_vec_size)
        
        # self.emb = nn.Embedding(input_size, word_vec_size)
        self.rnn = nn.LSTM(
            input_size=self.word_vec_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            dropout=dropout_p,
            batch_first=True,
            bidirectional=True,
        )

        self.fc1 = nn.Linear(hidden_size * 2, 300)
        # self.fc2 = nn.Linear(300, 300)
        # self.fc3 = nn.Linear(300, 300)
        self.fc4 = nn.Linear(300, n_classes)

        self.dropout = nn.Dropout(dropout_p)
        
        # We use LogSoftmax + NLLLoss instead of Softmax + CrossEntropy
        # self.activation = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        # |x| = (batch_size, length)
        x = self.emb(x).float()

        # |x| = (batch_size, length, word_vec_size)
        x, _ = self.rnn(x)
        
        # |x| = (batch_size, length, hidden_size * 2)
        # y = self.activation(self.fc(x[:, -1]))
        # |y| = (batch_size, n_classes)

        # y = self.fc(x[:, -1])
        y = self.fc1(self.dropout(x[:, -1]))
        # y = self.fc2(self.dropout(F.relu(y)))
        # y = self.fc3(self.dropout(F.relu(y)))
        y = self.fc4(self.dropout(F.relu(y)))

        return y