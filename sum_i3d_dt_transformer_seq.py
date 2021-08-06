#!/usr/bin/env python
# coding: utf-8

# In[1]:


#
#
# ===================== 
# Training a Classifier
# =====================
# 
#

import time, os, copy, math, numpy as np

import torch, torchvision
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.nn as nn
from torch.nn import Parameter, init
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import sys
import pickle
from collections import defaultdict
import random

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print(device)
### Get object features, distances and frame labels
from PIL import Image
object_classes = {'bottle': 0, 'bowl': 1, 'cheese': 2, 'cucumber': 3, 'knife': 4, 'lettuce': 5,
                  'peeler': 6, 'spoon': 7, 'tomato': 8, 'hand': 9}

class FrameFeatures(Dataset):
    features = None
    def __init__(self, object_feature_dir, dt_feature_dir, i3d_feature_dir, sequence_ids, test=False, transform=None):
        self.feature_sequences = []
        self.label_sequences = []
        self.dt_sequences = []
        self.i3d_sequences = []
        self.action_classes = {'SIL':0,'cut_fruit':1,'put_fruit2bowl':2,'peel_fruit':3,'stir_fruit':4,
                               'crack_egg':5,'add_saltnpepper':6,'stir_egg':7,'pour_oil':8,
                               'pour_egg2pan':9,'stirfry_egg':10,'take_plate':11,'put_egg2plate':12,
                               'pour_coffee':13,'pour_milk':14,'spoon_powder':15,'stir_milk':16,
                               'pour_cereals':17,'stir_cereals':18,'pour_flour':19,'stir_dough':20,
                               'pour_dough2pan':21,'fry_pancake':22,'put_pancake2plate':23,
                               'add_teabag':24,'pour_water':25,'cut_orange':26,'squeeze_orange':27,
                               'take_glass':28,'pour_juice':29,'fry_egg':30,'cut_bun':31,
                               'take_butter':32,'smear_butter':33,'put_toppingOnTop':34,
                               'put_bunTogether':35,'spoon_flour':36,'butter_pan':37,'take_eggs':38,
                               'take_cup':39,'pour_sugar':40,'stir_coffee':41,'take_bowl':42,
                               'take_knife':43,'spoon_sugar':44,'take_topping':45,'take_squeezer':46,
                               'stir_tea':47}
        self.transform = transform
        
        for sequence_id in sequence_ids:
            print(sequence_id)
            feature_sequences_video = []
            label_sequences_video = []
            dt_sequences_video = []
            i3d_sequences_video = []
            sequence_file = os.path.join(object_feature_dir, sequence_id.strip()+'.pkl')
            dt_sequence_file = os.path.join(dt_feature_dir, sequence_id.strip()+'.npy')
            i3d_sequence_file = os.path.join(i3d_feature_dir, sequence_id.strip()+'.npy')
            if os.path.exists(sequence_file) and os.path.exists(dt_sequence_file) \
                and os.path.exists(i3d_sequence_file):
                
                with open(sequence_file,'rb') as f1:
                    try:
                        sequence_dict = pickle.load(f1)
                        dt_matrix = np.load(dt_sequence_file)
                        i3d_feat = np.load(i3d_sequence_file)
                    except:
                        continue
                    prev_frame = sorted(list(sequence_dict.keys()))[0]
                    starts = []
                    stops = []
                    starts.append(prev_frame)
                    for frame in sorted(list(sequence_dict.keys())):
                        if sequence_dict[frame]['action'] != sequence_dict[prev_frame]['action']:
                            stops.append(prev_frame)
                            starts.append(frame)
                        prev_frame = frame
                    last_frame = min(dt_matrix.shape[0],len(i3d_feat),sorted(list(sequence_dict.keys()))[-1])
                    stops.append(last_frame)
                    feature_sequence = []
                    label_sequence = []
                    dt_sequence = []
                    i3d_sequence = []
                    sequence_count = 0
                    
                    for start, stop in zip(starts,stops):
                        if stop - start < 30:
                            continue
                        for frame_num in range(start,stop-14):
                            frame_feat = sequence_dict[frame_num]
                            features_frame = frame_feat['features']
                            if features_frame:
                                features_frame = features_frame[0]
                            object_labels = frame_feat['labels']
                            if object_labels:
                                object_labels = object_labels[0]
                            #print(object_labels)
                            if len(features_frame) != 0 and 'person' in object_labels:
                                action_label = sequence_dict[frame_num]['action']
                                action_label = self.action_classes[action_label]
                                sequence_count += 1
                                
                                dt_feature_frame = dt_matrix[frame_num-1,:]                    
                                i3d_feature_frame = i3d_feat[frame_num-1]
                                feature = self.__get_feature_pairs(features_frame, object_labels)      
                                if len(feature) != 0 or dt_feature_frame.size != 0:
                                    feature_sequence.append(feature)
                                    label_sequence.append(action_label)
                                    dt_sequence.append(dt_feature_frame)
                                    i3d_sequence.append(i3d_feature_frame)
                        if len(dt_sequence) != 0 or len(feature_sequence) != 0 or len(feature_sequence) != 0:
                            # print(len(i3d_sequence))
                            # print(len(feature_sequence))
                            observation = min(15, len(dt_sequence), len(feature_sequence), len(feature_sequence))
                            feature_sequences_video.append(feature_sequence[-observation:])
                            label_sequences_video.append(label_sequence[-observation:])
                            dt_sequences_video.append(dt_sequence[-observation:])
                            i3d_sequences_video.append(i3d_sequence[-observation:])
                            feature_sequence = []
                            label_sequence = []
                            dt_sequence = []
                            i3d_sequence = []   
                    feature_sequences_video = feature_sequences_video[:-1]
                    dt_sequences_video = dt_sequences_video[:-1]
                    i3d_sequences_video = i3d_sequences_video[:-1]
                    label_sequences_video = label_sequences_video[1:]
                    for feature_seq, label_seq, dt_seq, i3d_seq in\
                        zip(feature_sequences_video, label_sequences_video, \
                        dt_sequences_video, i3d_sequences_video):
                        
                        self.feature_sequences.append(feature_seq)
                        self.label_sequences.append(label_seq)
                        self.dt_sequences.append(dt_seq)
                        self.i3d_sequences.append(i3d_seq)
                        
    def __get_feature_pairs(self, features_frame, object_labels):
        feature_pairs_frame = []
        person_feature = features_frame[object_labels.index('person')]
        object_features = [x for i,x in enumerate(features_frame) if i!=object_labels.index('person')]
        #for person_feature in person_features:
            #print(len(person_feature[0]))
        for object_feature in object_features:
            #print(len(object_feature[0]))
            concat_features = np.concatenate((np.array(person_feature[0]), np.array(object_feature[0])))
            #print(concat_features.shape)
            feature_pairs_frame.append(concat_features)
        return feature_pairs_frame
        
    def __getitem__(self, index):
        feature_seq = self.feature_sequences[index]
        label_seq = self.label_sequences[index]
        dt_seq = self.dt_sequences[index]
        i3d_seq = self.i3d_sequences[index]

        return feature_seq, dt_seq, i3d_seq, label_seq
    
    
    def __len__(self):
        return len(self.label_sequences)        
        
object_feature_dir = '/home/roy/breakfast_features/'
dt_feature_dir = '/home/roy/breakfast_dtfv_features/'
i3d_feature_dir = '/home/roy/breakfast_i3dfeatures/'


# train_set_file = 'train_set_seq_1s.pkl'
# if os.path.exists(train_set_file):
    # training_set = pickle.load(open(train_set_file,'rb'))
# else:
    # training_set = FrameFeatures(object_feature_dir, dt_feature_dir, i3d_feature_dir, sequence_ids_train[:], test=False)
    # with open(train_set_file,'wb') as pkl:
        # pickle.dump(training_set, pkl, pickle.HIGHEST_PROTOCOL)
# print('{} training instances.'.format(len(training_set)))

# test_set_file = 'test_set_seq_1s.pkl'
# if os.path.exists(test_set_file):
    # test_set = pickle.load(open(test_set_file,'rb'))
# else:
    # test_set = FrameFeatures(object_feature_dir, dt_feature_dir, i3d_feature_dir, sequence_ids_test[:], test=True)
    # with open(test_set_file,'wb') as pkl:
        # pickle.dump(test_set, pkl, pickle.HIGHEST_PROTOCOL)
# print('{} test instances.'.format(len(test_set)))


# In[11]:


# ### Network definition

class SummationObj(nn.Module):
    def __init__(self, feature_length, embedding_length):
        super(SummationObj, self).__init__()

        """
        Arguments
        ---------
        output_size : 18
        feature_length : 512
       
        --------

        """
        
        self.feature_length = feature_length
        self.embedding_length = embedding_length
        self.dropout = nn.Dropout(0.5)
        self.W = nn.Linear(feature_length, self.embedding_length)

    def forward(self, input1, batch_size=None):
        weighted_out = self.W(input1)
        weighted_out = F.relu(weighted_out)
        #print(weighted_out.shape)
        summed_out = torch.sum(weighted_out, dim=1)
        summed_out = F.relu(summed_out)
        summed_out = self.dropout(summed_out)
        
        return summed_out

class SummationDTResnet(nn.Module):
    def __init__(self, feature_length, embedding_length):
        super(SummationDTResnet, self).__init__()

        """
        Arguments
        ---------
        output_size : 18
        feature_length : 512
       
        --------

        """
        
        self.feature_length = feature_length
        self.embedding_length = embedding_length
        self.dropout = nn.Dropout(0.5)
        self.W = nn.Linear(feature_length, self.embedding_length)

    def forward(self, input, batch_size=None):
        weighted_out = self.W(input)
        weighted_out = F.relu(weighted_out)
        summed_out = self.dropout(weighted_out)
        return summed_out

# In[12]:

class EncoderTransformer(nn.Module):
    def __init__(self, embedding_length, hidden_size, output_size, attention_heads):
        super(EncoderTransformer, self).__init__()
        self.input_size = embedding_length
        self.hidden_size = hidden_size
        self.output_size = output_size
        obj_feature_length = 512
        dt_feature_length = 64
        i3d_feature_length = 2048
        self.obj_embedding = SummationObj(obj_feature_length, 256)
        self.dt_embedding = SummationDTResnet(dt_feature_length, 64)
        self.i3d_embedding = SummationDTResnet(i3d_feature_length, 512)
        obj_encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=attention_heads, 
            dim_feedforward=32)
        self.obj_encoder = nn.TransformerEncoder(obj_encoder_layer, 2)
        obj_decoder_layer = nn.TransformerDecoderLayer(d_model=256, nhead=attention_heads)
        self.obj_decoder = nn.TransformerDecoder(obj_decoder_layer, 2)
        self.obj_fc = nn.Linear(256, self.output_size)
        
        dt_encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=attention_heads, 
            dim_feedforward=64)
        self.dt_encoder = nn.TransformerEncoder(dt_encoder_layer, 2)
        dt_decoder_layer = nn.TransformerDecoderLayer(d_model=64, nhead=attention_heads, 
            dim_feedforward=64)
        self.dt_decoder = nn.TransformerDecoder(dt_decoder_layer, 2)
        self.dt_fc = nn.Linear(64, self.output_size)
        
        i3d_encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=attention_heads, 
            dim_feedforward=self.hidden_size)
        self.i3d_encoder = nn.TransformerEncoder(i3d_encoder_layer, 2)
        i3d_decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=attention_heads)
        self.i3d_decoder = nn.TransformerDecoder(i3d_decoder_layer, 2)
        self.i3d_fc = nn.Linear(512, self.output_size)
        
        
    def forward(self, obj_sequence, dt_sequence, i3d_sequence):
        obj_embedded = []
        for input1 in obj_sequence:
            obj_embedded.append(self.obj_embedding(input1))
        obj_embedded = torch.stack(obj_embedded).permute(1,0,2)*math.sqrt(256)
        dt_embedded = self.dt_embedding(dt_sequence).permute(1,0,2)*math.sqrt(64)
        i3d_embedded = self.i3d_embedding(i3d_sequence).permute(1,0,2)*math.sqrt(512)
        
        obj_out = self.obj_encoder(obj_embedded)
        dt_out = self.dt_encoder(dt_embedded)
        i3d_out = self.i3d_encoder(i3d_embedded)
        
        #print(obj_out.shape)
        min_seq_len = min(obj_out.shape[1], dt_out.shape[1], i3d_out.shape[1])
        
        #total_out = torch.cat([obj_out[:,:min_seq_len,:], dt_out[:,:min_seq_len,:], i3d_out[:,:min_seq_len,:]], dim=2)
        obj_target = torch.rand(1,min_seq_len,256).to(device)
        obj_out = self.obj_decoder(obj_target, obj_out[:,:min_seq_len,:])
        obj_out = self.obj_fc(obj_out).squeeze()
        
        dt_target = torch.rand(1,min_seq_len,64).to(device)
        dt_out = self.dt_decoder(dt_target, dt_out[:,:min_seq_len,:])
        dt_out = self.dt_fc(dt_out).squeeze()
        
        i3d_target = torch.rand(1,min_seq_len,512).to(device)
        i3d_out = self.i3d_decoder(i3d_target, i3d_out[:,:min_seq_len,:])
        i3d_out = self.i3d_fc(i3d_out).squeeze()

        out = obj_out + dt_out + i3d_out
        return out

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class TrainTest():
    
    def __init__(self, model, trainset, testset, criterion, batch_size, nepoch, ckpt_path):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=0.00001)
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
            best_acc = self.state['acc']
            start_epoch = self.state['epoch']
            print('Epoch {}'.format(start_epoch))
            if start_epoch == self.nepoch:
                print('existing as epoch is max.')
            self.details = self.state['details']    
            self.best_acc = best_acc
            self.start_epoch = start_epoch + 1
            self.model.to(device)                    
        else:
            self.best_acc = -1.
            self.details = []   
            self.start_epoch = 0
    
    def test(self):
        correct = 0
        count = 0
        sequence = []
        with torch.no_grad():
            for i, data in enumerate(self.testloader, 0):        
                #print(len(data))
                obj_feat_seq = []
                dt_feat_seq = []
                i3d_feat_seq = []
                target_seq = []
                obj_sequence, dt_sequence, i3d_sequence, target_sequence = data    
                for obj_feats in obj_sequence: 
                    if len(obj_feats) == 0:
                        continue
                    obj_feats = torch.stack(obj_feats)
                    obj_feats = obj_feats.permute(1, 0, 2)
                    
                    obj_feat_seq.append(obj_feats.float().to(device))
                dt_feat_seq = torch.stack(dt_sequence).float().to(device)
                i3d_feat_seq = torch.stack(i3d_sequence).float().to(device)
                target_seq = torch.stack(target_sequence).squeeze().to(device)
                
                output = self.model(obj_feat_seq, dt_feat_seq, i3d_feat_seq)
                if output.shape[0] == 48:
                    output = output.unsqueeze(0)
                seq_length = min(output.shape[0],target_seq.shape[0])
                pred_seq = torch.argmax(output, dim=1)[:seq_length]
                target_seq = target_seq[:seq_length]
                correct = correct + torch.sum(pred_seq == target_seq).item()
            
                loss = self.criterion(output[:seq_length], target_seq)         
                #print(len(self.testloader))
                print("\rIteration: {}/{}, Loss: {}.".format(i+1, len(self.testloader), loss), end="")
                sys.stdout.flush()
                count += output.shape[0]
        return correct/count*100., correct   
        
    def train(self):        
        for epoch in range(self.start_epoch,self.nepoch):  
            start_time = time.time()        
            running_loss = 0.0
            correct = 0.
            count = 0.
            total_loss = 0
            self.optimizer.zero_grad()   
            
            iterations = 0
            for i, data in enumerate(self.trainloader, 0):        
                obj_feat_seq = []
                dt_feat_seq = []
                i3d_feat_seq = []
                target_seq = []
                obj_sequence, dt_sequence, i3d_sequence, target_sequence = data    
                for obj_feats in obj_sequence: 
                    if len(obj_feats) == 0:
                        continue
                    obj_feats = torch.stack(obj_feats)
                    obj_feats = obj_feats.permute(1, 0, 2)
                    
                    obj_feat_seq.append(obj_feats.float().to(device))
                dt_feat_seq = torch.stack(dt_sequence).float().to(device)
                i3d_feat_seq = torch.stack(i3d_sequence).float().to(device)
                target_seq = torch.stack(target_sequence).squeeze().to(device)
                
                output = self.model(obj_feat_seq, dt_feat_seq, i3d_feat_seq)
                #print(output.shape)
                if output.shape[0] == 48:
                    output = output.unsqueeze(0)
                with torch.no_grad():
                    seq_length = min(output.shape[0],target_seq.shape[0])
                    pred_seq = torch.argmax(output, dim=1)[:seq_length]
                    target_seq = target_seq[:seq_length]
                    correct = correct + torch.sum(pred_seq == target_seq).item()
                       
                
                loss = self.criterion(output[:seq_length], target_seq)         
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                count += output.shape[0]
                if i % batch_size == 0 and i>1:
                    print("\rIteration: {}/{}, Loss: {}.".format(i+1, len(self.trainloader), loss), end="")
                    self.optimizer.step()
                    self.optimizer.zero_grad()                    
                    running_loss = running_loss + loss
                    iterations += 1
            TRAIN_LOSS =  running_loss/iterations
            TRAIN_ACC = correct/count*100
            TEST_ACC, TEST_COUNT = self.test()
            self.details.append((TRAIN_LOSS,TRAIN_ACC,0.,TEST_ACC))
            #utils.draw_Fig(self.details)
            #plt.savefig('results/sum_classifier.png')        
            #plt.close()
            if TEST_ACC > self.best_acc:                
                self.state = {
                    'model': self.model.state_dict(),
                    'acc': TEST_ACC,
                    'epoch': epoch,
                    'details':self.details,            
                }        
                torch.save(self.state, self.chkpath)
                self.best_acc = TEST_ACC
            else:
                self.state['epoch'] = epoch
                torch.save(self.state, self.chkpath)
            elapsed_time = time.time() - start_time
            print('[{}] [{:.1f}] [Loss {:.3f}] [Correct : {}] [Trn. Acc {:.1f}] '.format(epoch, elapsed_time,
                    TRAIN_LOSS, correct,TRAIN_ACC),end=" ")
            print('[Test Cor {}] [Acc {:.1f}]'.format(TEST_COUNT,TEST_ACC))


# In[17]:


# ### define hyperparameters

embedding_length = 1024
dropout = 0.5
output_size = 48 # number of classes
hidden_size = 64
attention_heads = 1

#instantiate the model
model_ft = EncoderTransformer(embedding_length, hidden_size, output_size, attention_heads)
model_dict = model_ft.state_dict()

ckpt_sum = 'ckpt/sum_transformer_1s.pt'
pretrained_dict = torch.load(ckpt_sum)['model']
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
model_dict.update(pretrained_dict)
model_ft.load_state_dict(model_dict)


ckpt_i3d = 'ckpt/i3d_transformer_1s.pt'
pretrained_dict = torch.load(ckpt_i3d)['model']
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
model_dict.update(pretrained_dict)
model_ft.load_state_dict(model_dict)


ckpt_dt = 'ckpt/dt_transformer_1s.pt'
pretrained_dict = torch.load(ckpt_dt)['model']
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
model_dict.update(pretrained_dict)
model_ft.load_state_dict(model_dict)


ckpt_path = 'ckpt/sum_i3d_dt_transformer_1s.pt'    

criterion = nn.CrossEntropyLoss()

batch_size = 8

sequence_ids_test = open('test.s1').readlines()
test_set = FrameFeatures(object_feature_dir, dt_feature_dir, i3d_feature_dir, sequence_ids_test[:10], test=True)
print('{} test instances.'.format(len(test_set)))

# sequence_ids_train = sorted(open('train.s1').readlines())

# nepochs = 10
# training_set = FrameFeatures(object_feature_dir, dt_feature_dir, i3d_feature_dir, sequence_ids_train[:], test=False)
# print('{} training instances.'.format(len(training_set)))
# EXEC = TrainTest(model_ft, training_set, test_set, criterion, batch_size, nepochs, ckpt_path)
# EXEC.train()

from sklearn.metrics import classification_report
pred_labels = []
target_labels = []
testloader = torch.utils.data.DataLoader(test_set, batch_size=1,
                                          shuffle=False, num_workers=0)
pred_labels = []
target_labels = []
model_ft.to(device)
with torch.no_grad():
    for i, data in enumerate(testloader, 0):        
        #print(len(data))
        obj_feat_seq = []
        dt_feat_seq = []
        i3d_feat_seq = []
        target_seq = []
        obj_sequence, dt_sequence, i3d_sequence, target_sequence = data    
        for obj_feats in obj_sequence: 
            if len(obj_feats) == 0:
                continue
            obj_feats = torch.stack(obj_feats)
            obj_feats = obj_feats.permute(1, 0, 2)
            
            obj_feat_seq.append(obj_feats.float().to(device))
        dt_feat_seq = torch.stack(dt_sequence).float().to(device)
        i3d_feat_seq = torch.stack(i3d_sequence).float().to(device)

        target_seq = torch.stack(target_sequence).squeeze().to(device)
                
        output = model_ft(obj_feat_seq, dt_feat_seq, i3d_feat_seq) 
        if output.shape[0] == 48:
            output = output.unsqueeze(0)
        seq_length = min(output.shape[0],target_seq.shape[0])
        pred_seq = torch.argmax(output, dim=1)[:seq_length]
        target_seq = target_seq[:seq_length]
        #print(target_seq)
        #print(pred_seq)
        pred_labels.append(pred_seq)
        target_labels.append(target_seq)
    
pred_mod = []
target_mod = []
for preds, targets in zip(pred_labels, target_labels):
    pred_mod.append(preds[0].item())
    if targets.dim() == 0:
        targets = torch.unsqueeze(targets,0)
    target_mod.append(targets[0].item())
print(classification_report(target_mod, pred_mod,digits=4))