import torch
import torch.nn as nn

class EncoderCondBi(nn.Module):

    def __init__(self, seq_len, n_features=1, embedding_dim=64, condition_dim=11):
        super(EncoderCondBi, self).__init__()
        self.seq_len, self.n_features = seq_len, n_features
        self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim
        self.condition = condition_dim

        self.rnn1 = nn.LSTM(
            input_size=n_features + condition_dim,  # Zmodyfikowane input_size
            hidden_size=self.embedding_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.rnn2 = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.embedding_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )


    def forward(self, x, condition):
        # Połączenie wektora warunkowego z danymi wejściowymi
        x = torch.cat((x, condition), dim=2)
        x, (_, _) = self.rnn1(x)
        x, (hidden_n, _) = self.rnn2(x)
        
        hidden_n = hidden_n.view(self.rnn2.num_layers, 2, x.size(0), self.embedding_dim)
        hidden_n = torch.cat((hidden_n[:, 0, :, :], hidden_n[:, 1, :, :]), dim=2)
        
        return hidden_n.squeeze(0)


class DecoderCondBi(nn.Module):

    def __init__(self, seq_len, input_dim=64, n_features=1, condition_dim=11):
        super(DecoderCondBi, self).__init__()
        self.seq_len, self.input_dim = seq_len, input_dim
        self.hidden_dim, self.n_features = input_dim, n_features
        self.condition_dim = condition_dim

        self.rnn1 = nn.LSTM(
            input_size=self.input_dim + condition_dim,  # Zmodyfikowane input_size
            hidden_size=self.input_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.rnn2 = nn.LSTM(
            input_size=2 * self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        self.output_layer = nn.Linear(2* (self.hidden_dim), n_features)


    def forward(self, x, condition):
        # Połączenie wektora warunkowego z danymi wejściowymi
        x = x.unsqueeze(1).repeat(1, self.seq_len, 1) #Reapeat vector
        x = torch.cat((x, condition), dim=2)
        
        x, (_, _) = self.rnn1(x)
        x, (_, _) = self.rnn2(x)
        out = self.output_layer(x)
        return out


class RecurrentAutoencoderCondBi(nn.Module):
    
    def __init__(self, seq_len, n_features, embedding_dim=64, condition_dim=10, device=None):
        super(RecurrentAutoencoderCondBi, self).__init__()
        self.encoder = EncoderCondBi(seq_len, n_features, embedding_dim, condition_dim).to(device)
        self.decoder = DecoderCondBi(seq_len, embedding_dim * 2, n_features, condition_dim).to(device)


    def forward(self, x, condition):
        x = self.encoder(x, condition)
        x = self.decoder(x, condition)
        return x