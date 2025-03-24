import torch
import torch.nn as nn

class EncoderCond(nn.Module):

    def __init__(self, seq_len, n_features=1, embedding_dim=64, condition_dim=11):
        super(EncoderCond, self).__init__()
        self.seq_len, self.n_features = seq_len, n_features
        self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim
        self.condition = condition_dim

        self.rnn1 = nn.LSTM(
            input_size=n_features + condition_dim,  # Zmodyfikowane input_size
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
        )
        self.rnn2 = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=embedding_dim,
            num_layers=1,
            batch_first=True
        )


    def forward(self, x, condition):
        # Połączenie wektora warunkowego z danymi wejściowymi
        x = torch.cat((x, condition), dim=2)
        x, (_, _) = self.rnn1(x)
        x, (hidden_n, _) = self.rnn2(x)
        return hidden_n.squeeze(0)


class DecoderCond(nn.Module):

    def __init__(self, seq_len, input_dim=64, n_features=1, condition_dim=11):
        super(DecoderCond, self).__init__()
        self.seq_len, self.input_dim = seq_len, input_dim
        self.hidden_dim, self.n_features = 2 * input_dim, n_features
        self.condition_dim = condition_dim

        self.rnn1 = nn.LSTM(
            input_size=input_dim + condition_dim,  # Zmodyfikowane input_size
            hidden_size=input_dim,
            num_layers=1,
            batch_first=True
        )
        self.rnn2 = nn.LSTM(
            input_size=input_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
        )
        self.output_layer = nn.Linear(self.hidden_dim, n_features)


    def forward(self, x, condition):
        # Połączenie wektora warunkowego z danymi wejściowymi
        x = x.unsqueeze(1).repeat(1, self.seq_len, 1) #Reapeat vector
        x = torch.cat((x, condition), dim=2)
        
        x, (_, _) = self.rnn1(x)
        x, (_, _) = self.rnn2(x)
        out = self.output_layer(x)
        return out


class RecurrentAutoencoderCond(nn.Module):
    
    def __init__(self, seq_len, n_features, embedding_dim=64, condition_dim=10, device=None):
        super(RecurrentAutoencoderCond, self).__init__()
        self.encoder = EncoderCond(seq_len, n_features, embedding_dim, condition_dim).to(device)
        self.decoder = DecoderCond(seq_len, embedding_dim, n_features, condition_dim).to(device)


    def forward(self, x, condition):
        x = self.encoder(x, condition)
        x = self.decoder(x, condition)
        return x