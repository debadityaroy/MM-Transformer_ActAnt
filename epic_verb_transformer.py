#
# =====================
# Training a Classifier
# =====================
# 
 

import time, os, copy, numpy as np

import torch, torchvision
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import sys
import pickle
import numpy as np
from torch.nn import Parameter, init
import torch.nn.functional as F
import math
from PIL import Image
from collections import defaultdict
import pickle

from collections import defaultdict
import random
import pandas as pd

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

action_annotation = pd.read_csv('EPIC_train_action_labels.txt')

verbs = list(action_annotation['verb_class'].values)
nouns = list(action_annotation['noun_class'].values)
actions_dict = {}
count = 0
for verb, noun in zip(verbs, nouns):
    action = str(verb)+'_'+str(noun)
    if action not in actions_dict.keys():
        actions_dict[str(verb)+'_'+str(noun)] = count
        count += 1
    
SEQUENCE_SIZE = 10
NUM_OBJECTS = 2

class I3D_EPICval(Dataset):
    def __init__(self, ann_file, obj_feat_dir, person_feat_dir, flow_feat_dir, video_ids, transform=None):
        self.obj_feat_list = []
        self.verbs_list = []
        self.nouns_list = []
        self.person_feat_list = []
        self.flow_feat_list = []
        
        
        action_annotation = pd.read_csv(ann_file)
        # [VID, Obs_start, Obs_end, Obs_noun, Obs_verb, Fut_start, Fut_end, Fut_noun,Fut_verb]
        
        
        for video_id in video_ids:
            video_id = video_id.strip('\n')
            starts = list(action_annotation.loc[action_annotation['video_id'] == video_id]['start_frame'].values)[:-1]
            stops = list(action_annotation.loc[action_annotation['video_id'] == video_id]['stop_frame'].values)[:-1]
            verbs = list(action_annotation.loc[action_annotation['video_id'] == video_id]['verb_class'].values)[1:]
            nouns = list(action_annotation.loc[action_annotation['video_id'] == video_id]['noun_class'].values)[1:]
            
            try:
                obj_feat_dict = pickle.load(open(os.path.join(obj_feat_dir, video_id+'.pkl'),'rb'))
                person_feat_dict = pickle.load(open(os.path.join(person_feat_dir, video_id+'.pkl'),'rb'))
                flow_feat = np.load(os.path.join(flow_feat_dir, video_id+'-flow.npz'))['feature']
                #print(flow_feat.shape)
            except:
                print('{}'.format(video_id))
                continue
            obj_feat_in_video = []    
            verbs_in_video = []
            nouns_in_video = []
            person_feat_in_video = []
            flow_feat_in_video = []
            for start, stop, verb, noun in zip(starts, stops, verbs, nouns):
                feat_frames = []
                for frame_num in range(int(start),int(stop-60)):
                    if frame_num%6==0 and (frame_num/6) in obj_feat_dict.keys():
                        feat_frames.append(frame_num/6)
                #print(feat_frames)
                if SEQUENCE_SIZE > len(feat_frames):
                    continue
                else:
                    start_seq_num = -SEQUENCE_SIZE
                
                obj_feat_in_seg = [] 
                person_feat_in_seg = []
                flow_feat_in_seg = []
                verbs_in_seg = []
                nouns_in_seg = []
                for feat_frame in feat_frames[start_seq_num:]:
                    obj_feat = obj_feat_dict[feat_frame]
                    if len(obj_feat) != 0 and feat_frame < flow_feat.shape[1]:
                        flow_feat_in_seg.append(flow_feat[:,int(feat_frame)-1,:])
                        verbs_in_seg.append(verb)
                        nouns_in_seg.append(int(noun)-1)
                        if len(obj_feat) >= NUM_OBJECTS:
                            obj_feat = obj_feat[:NUM_OBJECTS]
                            #print(obj_feat.shape)
                        else:
                            obj_feat = torch.stack([obj_feat, torch.zeros_like(obj_feat)]).squeeze(1)
                            #print(obj_feat.shape)
                        obj_feat_in_seg.append(obj_feat) 
                    if feat_frame in person_feat_dict.keys():
                        person_feat_in_seg.append(person_feat_dict[feat_frame][0]['feature'][0])
                    else:
                        # padding in case there is no person in the frame
                        person_feat_in_seg.append([0.0 for _ in range(256)]) 
                    
                    
                if obj_feat_in_seg and person_feat_in_seg:
                    obj_feat_in_video.append(obj_feat_in_seg)
                    person_feat_in_video.append(person_feat_in_seg)
                    flow_feat_in_video.append(flow_feat_in_seg)
                    verbs_in_video.append(verbs_in_seg)
                    nouns_in_video.append(nouns_in_seg)
                  
            self.obj_feat_list.extend(obj_feat_in_video)
            self.person_feat_list.extend(person_feat_in_video)
            self.flow_feat_list.extend(flow_feat_in_video)
            self.verbs_list.extend(verbs_in_video)
            self.nouns_list.extend(nouns_in_video) 
  
            
    def __getitem__(self, index):

        obj_feat_seq = self.obj_feat_list[index]
        person_feat_seq = self.person_feat_list[index]  
        flow_feat_seq = self.flow_feat_list[index]
        verb_seq = self.verbs_list[index]
        noun_seq = self.nouns_list[index]
        
        return obj_feat_seq, person_feat_seq, flow_feat_seq, verb_seq, noun_seq

    def __len__(self):
        return len(self.verbs_list)


from torch.nn import Parameter, init
import torch.nn.functional as F

class SummationDTResnet(nn.Module):
    def __init__(self, feature_length, embedding_length):
        super(SummationDTResnet, self).__init__()
        """
        Arguments
        ---------
        output_size : 18
        feature_length : 512
        ---------
        """
        self.feature_length = feature_length
        self.embedding_length = embedding_length
        self.dropout = nn.Dropout(0.5)
        self.W = nn.Linear(feature_length, self.embedding_length)

    def forward(self, input, batch_size=None):
        weighted_out = self.W(input)
        weighted_out = F.relu(weighted_out)
        out = self.dropout(weighted_out)
        return out

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class EncoderTransformer(nn.Module):
    def __init__(self, embedding_length, verb_size, noun_size, attention_heads):
        super(EncoderTransformer, self).__init__()
        self.verb_size = verb_size
        self.noun_size = noun_size
        self.flow_embedding = SummationDTResnet(1024, 512)
        self.flow_pos_encoder = PositionalEncoding(512)
        flow_encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=attention_heads)
        self.flow_encoder = nn.TransformerEncoder(flow_encoder_layer, 2)
        flow_decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=attention_heads)
        self.flow_decoder = nn.TransformerDecoder(flow_decoder_layer, 2)
        self.flow_fc = nn.Linear(512, self.verb_size)
             
        
    def forward(self, obj_feat_seq, person_feat_seq, flow_feat_seq):
        flow_feat_seq = torch.stack(flow_feat_seq).squeeze(2)
        #print(flow_feat_seq.shape)
        flow_embedded = self.flow_embedding(flow_feat_seq).permute(1,0,2)*math.sqrt(512)
        flow_embedded = self.flow_pos_encoder(flow_embedded)
        flow_out = self.flow_encoder(flow_embedded)
        
        seq_len = flow_out.shape[1]
        target = torch.rand(1,seq_len,512).to(device)
        #print(target.shape)
        verb_out = self.flow_decoder(target, flow_out)
        verb_out = self.flow_fc(verb_out).squeeze()
        
        return verb_out

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

        
class TrainTest():
    
    def __init__(self, model, trainset, testset, criterion, batch_size, nepoch, ckpt_path):
        self.model = model
        self.optimizer = optim.Adam(model.parameters())
        #self.optimizer = optim.SGD(model.parameters(), lr=0.0005, weight_decay=0.9 )
        self.criterion = criterion
        self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=1,
                                          shuffle=True, num_workers=0)
        self.testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                          shuffle=False, num_workers=0)
          
        self.model.to(device)          
        self.chkpath = ckpt_path
        self.batch_size = batch_size
        self.nepoch = nepoch
        self.trainset_size = len(trainset)
        if not os.path.exists('ckpt/'):
            os.mkdir('ckpt/')            
        print(self.chkpath)
        if os.path.exists(self.chkpath) == True:
            print('load from ckpt', end=' ')
            self.state = torch.load(self.chkpath)
            self.model.load_state_dict(self.state['model'])
            best_verb_acc = self.state['verb_acc']
            start_epoch = self.state['epoch']
            print('Epoch {}'.format(start_epoch))
            if start_epoch == self.nepoch:
                print('exiting as epoch is max.')
            self.details = self.state['details']    
            self.best_verb_acc = best_verb_acc
            self.start_epoch = start_epoch + 1
            self.model.to(device)                    
        else:
            self.best_verb_acc = -1.
            self.details = []   
            self.start_epoch = 0
            
        
    def test(self):
        correct_verb = 0
        correct_noun = 0
        count = 0
        sequence = []
        with torch.no_grad():
            for i, data in enumerate(self.testloader, 0):        
                obj_feat_seq, person_feat_seq, flow_feat_seq, verb_seq, noun_seq = data 
                #print(len(obj_feat_seq))
                
                obj_feat_seq = [obj_feat.to(device) for obj_feat in obj_feat_seq]
                # print(len(person_feat_seq))
                person_feat_seq = [torch.Tensor(person_feat).to(device) for person_feat in person_feat_seq]
                
                flow_feat_seq = [flow_feat.to(device) for flow_feat in flow_feat_seq]
                
                target_verb_seq = torch.Tensor(verb_seq).long().to(device)
                target_noun_seq = torch.Tensor(noun_seq).long().to(device)
                
                verb_out = self.model(obj_feat_seq, person_feat_seq, flow_feat_seq)
                pred_verb_seq = torch.argmax(verb_out, dim=1)
                #print(pred_verb_seq)
                
                correct_verb = correct_verb + torch.sum(pred_verb_seq==target_verb_seq).item()
                
                loss = self.criterion(verb_out, target_verb_seq)
               
                print("\rIteration: {}/{},  Loss{}.".format(i+1, len(self.testloader), loss), end="")
                sys.stdout.flush()
                count += target_verb_seq.shape[0]
        return correct_verb/count*100., correct_verb     
    

    def train(self):        
        for epoch in range(self.start_epoch,self.nepoch):  
            start_time = time.time()        
            running_loss = 0.0
            correct_verb = 0
            count = 0.
            total_loss = 0
            self.optimizer.zero_grad()   
            
            iterations = 0
            for i, data in enumerate(self.trainloader, 0):        
                obj_feat_seq, person_feat_seq, flow_feat_seq, verb_seq, noun_seq = data 
                #print(len(obj_feat_seq))
                
                obj_feat_seq = [obj_feat.to(device) for obj_feat in obj_feat_seq]
                # print(len(person_feat_seq))
                person_feat_seq = [torch.Tensor(person_feat).to(device) for person_feat in person_feat_seq]
                
                flow_feat_seq = [flow_feat.to(device) for flow_feat in flow_feat_seq]
                
                target_verb_seq = torch.Tensor(verb_seq).long().to(device)
                target_noun_seq = torch.Tensor(noun_seq).long().to(device)
                
                verb_out = self.model(obj_feat_seq, person_feat_seq, flow_feat_seq)
                # print(verb_out.shape)
                pred_verb_seq = torch.argmax(verb_out, dim=1)
                
                with torch.no_grad():
                    correct_verb = correct_verb + torch.sum(pred_verb_seq==target_verb_seq).item()
                    
                loss = self.criterion(verb_out, target_verb_seq)              
                
                loss.backward(retain_graph=True)
                
                running_loss += loss.item()
                count += target_verb_seq.shape[0]
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                if i % batch_size == 0 and i>1:
                    print("\rIteration: {}/{}, Loss: {}".format(i+1, len(self.trainloader), running_loss/i), end="")
                    self.optimizer.step()
                    self.optimizer.zero_grad() 
                    #torch.cuda.empty_cache()
                    
                    iterations += 1
                    
            TRAIN_LOSS =  running_loss/iterations
            TRAIN_VERB_ACC = correct_verb/count*100
            TEST_VERB_ACC, TEST_VERB_COUNT,  = self.test()
            self.details.append((TRAIN_LOSS, 0., TEST_VERB_ACC))

            if TEST_VERB_ACC > self.best_verb_acc:                
                self.state = {
                    'model': self.model.state_dict(),
                    'verb_acc': TEST_VERB_ACC,
                    'epoch': epoch,
                    'details':self.details,            
                }        
                torch.save(self.state, self.chkpath)
                self.best_verb_acc = TEST_VERB_ACC
                
            else:
                self.state['epoch'] = epoch
                torch.save(self.state, self.chkpath)
            elapsed_time = time.time() - start_time
            print('[{}] [{:.1f}] [Loss {:.3f}] [Verb Correct : {}] [Trn. Verb Acc {:.1f}] '.format(epoch, elapsed_time,
            TRAIN_LOSS, correct_verb, TRAIN_VERB_ACC),end=" ")
            print('[Test Verb Cor {}] [Verb Acc {:.1f}] '.format(TEST_VERB_COUNT, TEST_VERB_ACC))             


# In[17]:


# ### define hyperparameters

embedding_length = 128
dropout = 0.5
num_verb_classes = 125 
num_noun_classes = 351
attention_heads = 1

#instantiate the model
anticipation_model = EncoderTransformer(embedding_length, num_verb_classes, num_noun_classes, attention_heads)
ckpt_path = 'ckpt/i3d_flow_transformer_verb_1s.pt'
    

#define optimizer and loss
#class_weights_tensor = class_weights_tensor.float().to(device)
#criterion = nn.CrossEntropyLoss(weight = class_weights_tensor)
criterion = nn.CrossEntropyLoss()


#Loss Function
#Observe that all parameters are being optimized
#optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.001)
#optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.01, momentum=0.9, weight_decay=0.9 )

# Decay LR by a factor of 0.1 every 20 epochs
#exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.1)


batch_size = 8

obj_feat_dir = '/data/roy/graph/EPIC_KITCHENS_2020/obj_feat'
person_feat_dir = '/data/roy/graph/EPIC_KITCHENS_2020/person_maskrcnn_features'
flow_feat_dir = '/data/roy/graph/EPIC_KITCHENS_2020/i3d_flow_features'

sequence_ids_train = sorted(open('train.txt').readlines())
sequence_ids_test = sorted(open('test.txt').readlines())

train_ann_file = 'train_annotations.csv'
test_ann_file = 'test_annotations.csv'

test_set = I3D_EPICval(test_ann_file, obj_feat_dir, person_feat_dir, flow_feat_dir, sequence_ids_test[:])
print('{} test instances.'.format(len(test_set)))


nepochs = 10
train_set = I3D_EPICval(train_ann_file, obj_feat_dir, person_feat_dir, flow_feat_dir, sequence_ids_train[:])
print('{} train instances.'.format(len(train_set)))
EXEC = TrainTest(anticipation_model, train_set, test_set, criterion, batch_size, nepochs, ckpt_path)
EXEC.train()
