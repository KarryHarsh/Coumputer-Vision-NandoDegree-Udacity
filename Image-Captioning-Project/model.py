import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        self.hidden= hidden_size
        self.embeddings = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers,
                            dropout=0.5, batch_first=True)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.num_layers = num_layers
        #self.init_weights()
    
    def forward(self, features, captions):
        captions = captions[:, :-1]

        emends = self.embeddings(captions)
        
        # Concatenate the features and caption inputs
        features = features.unsqueeze(1)
        inputs = torch.cat((features, emends), 1)
        outputs, _ = self.lstm(inputs)
        outputs = self.dropout(outputs)
        # Convert LSTM outputs to word predictions
        outputs = self.fc(outputs)
        
        return outputs

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        tokens = []
        
        hidden = (torch.randn(1, 1, self.hidden).to(inputs.device),
                  torch.randn(1, 1, self.hidden).to(inputs.device))
        
        for i in range(max_len):
            out, states = self.lstm(inputs, hidden)
            out = self.fc(out.squeeze(1))
            _, predicted = out.max(1) 
            tokens.append(predicted.item())
            inputs = self.embeddings(predicted) 
            inputs = inputs.unsqueeze(1)
        return tokens