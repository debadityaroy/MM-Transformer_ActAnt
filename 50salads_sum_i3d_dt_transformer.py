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
import math


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# ## Get object features, distances and frame labels

# ### Helper functions


from PIL import Image
object_classes = {'bottle': 0, 'bowl': 1, 'cheese': 2, 'cucumber': 3, 'knife': 4, 
                  'lettuce': 5, 'peeler': 6, 'spoon': 7, 'tomato': 8, 'hand': 9}

from collections import defaultdict


# ### Debug features class

import pickle

from collections import defaultdict
import random

sequence_size = 30

class FrameFeatures(Dataset):
    features = None
    def __init__(self, object_feature_dir, dt_feature_dir, i3d_feature_dir, sequence_ids, transform=None):
        self.feature_sequences = []
        self.label_sequences = []
        self.displacement_sequences = []
        self.dt_sequences = []
        self.i3d_sequences = []
        self.action_classes = {'cut_cheese_': 0, 'place_cheese_into_bowl_': 1, 'add_dressing_': 2,
                               'cut_cucumber_': 3, 'add_vinegar_': 4, 'place_cucumber_into_bowl_': 5,
                               'place_tomato_into_bowl_': 6, 'add_salt_': 7, 'mix_dressing_': 8,
                               'place_lettuce_into_bowl_': 9, 'add_oil_': 10, 'cut_lettuce_': 11,
                               'cut_tomato_': 12, 'peel_cucumber_': 13, 'mix_ingredients_': 14,
                               'add_pepper_': 15, 'serve_salad_onto_plate_': 16, 'background': 17}
        self.transform = transform
        
        for sequence_id in sequence_ids:
            # print(sequence_id)
            feature_sequences_video = []
            label_sequences_video = []
            displacement_sequences_video = []
            dt_sequences_video = []
            i3d_sequences_video = []
            count_background = 0
            sequence_file = os.path.join(object_feature_dir, sequence_id.strip())
            dt_sequence_file = os.path.join(dt_feature_dir, sequence_id.strip())
            i3d_sequence_file = os.path.join(i3d_feature_dir, sequence_id.strip('.pkl\n')+'.npy')
            sequence_count = 0
            with open(sequence_file,'rb') as f1, open(dt_sequence_file,'rb') as f2:
                sequence_dict = pickle.load(f1)
                dt_dict = pickle.load(f2)
                i3d_feat = np.load(i3d_sequence_file)
                feature_sequence = []
                label_sequence = []
                displacement_sequence = []
                dt_sequence = []
                i3d_sequence = []
                sequence_count = 0
                last_frame = min(i3d_feat.shape[1], int(sorted(list(sequence_dict.keys()))[-1]))
                for frame_num in sequence_dict.keys():
                    frame_feat = sequence_dict[frame_num]
                    action_label = sequence_dict[frame_num]['action']
                    action_label = self.action_classes[action_label]
                    object_labels = frame_feat['labels']
                    features_frame = frame_feat['features']
                    # print(len(features_frame))
                    centers_frame = frame_feat['centers']
                    if frame_num not in dt_dict.keys():
                        continue
                    else:    
                        dt_feature_frame = dt_dict[frame_num]['feature']
                    i3d_feature_frame = i3d_feat[:,int(frame_num)-1]
                    #print(object_labels)
                    if len(features_frame) != 0:  
                        sequence_count += 1
                        label_sequence.append(action_label)
                        dt_sequence.append(dt_feature_frame)
                        i3d_sequence.append(i3d_feature_frame)
                        feature, displacement = self.__get_feature_pairs_and_displacement(features_frame, object_labels, centers_frame)
                        #print(len(feature))
                        feature_sequence.append(feature)
                        displacement_sequence.append(displacement)
                        if sequence_count % sequence_size == 0:
                            #print(len(feature_sequence))
                            feature_sequences_video.append(feature_sequence)
                            displacement_sequences_video.append(displacement_sequence)
                            label_sequences_video.append(label_sequence)
                            dt_sequences_video.append(dt_sequence)
                            i3d_sequences_video.append(i3d_sequence)
                            feature_sequence = []
                            label_sequence = []
                            displacement_sequence = []
                            dt_sequence = []
                            i3d_sequence = []
                feature_sequences_video = feature_sequences_video[:-2]
                label_sequences_video = label_sequences_video[2:]
                displacement_sequences_video = displacement_sequences_video[:-2]
                dt_sequences_video = dt_sequences_video[:-2]
                i3d_sequences_video = i3d_sequences_video[:-2]
                for feature_seq, label_seq, displacement_seq, dt_seq, i3d_seq in\
                    zip(feature_sequences_video, label_sequences_video, 
                    displacement_sequences_video, dt_sequences_video, i3d_sequences_video):
                    
                    self.feature_sequences.append(feature_seq)
                    self.label_sequences.append(label_seq)
                    self.displacement_sequences.append(displacement_seq)
                    self.dt_sequences.append(dt_seq)
                    self.i3d_sequences.append(i3d_seq)
    def __get_feature_pairs_and_displacement(self, features_frame, object_labels, centers_frame):
        feature_pairs_frame = []
        displacements_frame= []
        hand_feature = features_frame[object_labels.index('hand')]
        hand_center = centers_frame[object_labels.index('hand')]
        #print(len(hand_feature[0]))
        object_features = [x for i,x in enumerate(features_frame) if i!=object_labels.index('hand')]
        object_centers = [x for i,x in enumerate(centers_frame) if i!=object_labels.index('hand')]
        for object_feature, object_center in zip(object_features, object_centers):
            #print(len(object_feature[0]))
            concat_features = np.concatenate((np.array(hand_feature[0]), np.array(object_feature[0])))
            displacement = [hand_center[0] - object_center[0], hand_center[1] - object_center[1]]
            #print(concat_features.shape)
            feature_pairs_frame.append(concat_features)
            displacements_frame.append(displacement)
        return feature_pairs_frame, displacements_frame
        
    def __getitem__(self, index):
        feature_seq = self.feature_sequences[index]
        label_seq = self.label_sequences[index]
        displacement_seq = self.displacement_sequences[index]
        dt_seq = self.dt_sequences[index]
        i3d_seq = self.i3d_sequences[index]


        return feature_seq, displacement_seq, dt_seq, i3d_seq, label_seq
    
    
    def __len__(self):
        return len(self.label_sequences)



from torch.nn import Parameter, init
import torch.nn.functional as F

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
            #print(input1.shape)
        obj_embedded = torch.stack(obj_embedded).permute(1,0,2)*math.sqrt(256)
        dt_embedded = self.dt_embedding(dt_sequence).unsqueeze(0)*math.sqrt(64)
        i3d_embedded = self.i3d_embedding(i3d_sequence).permute(1,0,2)*math.sqrt(512)
        
        obj_out = self.obj_encoder(obj_embedded)
        dt_out = self.dt_encoder(dt_embedded)
        i3d_out = self.i3d_encoder(i3d_embedded)
        
        #print(obj_out.shape)
        min_seq_len = min(obj_out.shape[1], dt_out.shape[1], i3d_out.shape[1])
        
        #total_out = torch.cat([obj_out[:,:min_seq_len,:], dt_out[:,:min_seq_len,:], i3d_out[:,:min_seq_len,:]], dim=2)
        obj_target = torch.rand(1, min_seq_len, 256).to(device)
        obj_out = self.obj_decoder(obj_target, obj_out[:,:min_seq_len,:])
        obj_out = self.obj_fc(obj_out).squeeze()
        
        dt_target = torch.rand(1, min_seq_len, 64).to(device)
        dt_out = self.dt_decoder(dt_target, dt_out[:,:min_seq_len,:])
        dt_out = self.dt_fc(dt_out).squeeze()
        
        i3d_target = torch.rand(1, min_seq_len, 512).to(device)
        i3d_out = self.i3d_decoder(i3d_target, i3d_out[:,:min_seq_len,:])
        i3d_out = self.i3d_fc(i3d_out).squeeze()

        out = obj_out + dt_out + i3d_out
        return out

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class TrainTest():
    
    def __init__(self, model, trainset, testset, criterion, batch_size, nepoch, ckpt_path):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)
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
                displacement_seq = []
                obj_sequence, displacement_sequence, dt_sequence, i3d_sequence, label_sequence = data    
                for obj_feats, labels, dt_feats, i3d_feats, displacements in\
                    zip(obj_sequence, label_sequence, dt_sequence, i3d_sequence, displacement_sequence): 
                    if len(obj_feats) == 0:
                        continue
                    obj_feats = torch.stack(obj_feats)
                    obj_feats = obj_feats.permute(1, 0, 2)
                    
                    obj_feat_seq.append(obj_feats.float().to(device))
                    
                    dt_feats = [float(i[0]) for i in dt_feats]
                    dt_feat_seq.append(torch.Tensor(dt_feats).float().to(device))
                    i3d_feat_seq.append(torch.Tensor(i3d_feats).float().to(device))
                    target_seq.append(labels.long().to(device))
                    
                dt_feat_seq = torch.stack(dt_feat_seq).float().to(device)
                i3d_feat_seq= torch.stack(i3d_feat_seq).float().to(device)
                target_seq = torch.stack(target_seq).squeeze().to(device)
                # print(target_seq.shape)
                output = self.model(obj_feat_seq, dt_feat_seq, i3d_feat_seq)
                loss = self.criterion(output, target_seq)
                
                pred_seq = torch.argmax(output, dim=1)
                correct = correct + torch.sum(pred_seq==target_seq).item()                    
                
                print("\rIteration: {}/{}, Loss: {}.".format(i+1, len(self.testloader), loss), end="")
                sys.stdout.flush()
                count += sequence_size
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
                #print(len(data))
                obj_feat_seq = []
                dt_feat_seq = []
                i3d_feat_seq = []
                target_seq = []
                displacement_seq = []
                obj_sequence, displacement_sequence, dt_sequence, i3d_sequence, label_sequence = data    
                for obj_feats, labels, dt_feats, i3d_feats, displacements in\
                    zip(obj_sequence, label_sequence, dt_sequence, i3d_sequence, displacement_sequence): 
                    if len(obj_feats) == 0:
                        continue
                    obj_feats = torch.stack(obj_feats)
                    obj_feats = obj_feats.permute(1, 0, 2)
                    
                    obj_feat_seq.append(obj_feats.float().to(device))
                    dt_feats = [float(i[0]) for i in dt_feats]
                    dt_feat_seq.append(torch.Tensor(dt_feats))
                    i3d_feat_seq.append(torch.Tensor(i3d_feats))
                    target_seq.append(labels.long().to(device))
                
                dt_feat_seq = torch.stack(dt_feat_seq).float().to(device)
                i3d_feat_seq= torch.stack(i3d_feat_seq).float().to(device)
                target_seq = torch.stack(target_seq).squeeze().to(device)
                #print(target_seq.shape)
                output = self.model(obj_feat_seq, dt_feat_seq, i3d_feat_seq)
                pred_seq = torch.argmax(output, dim=1)

                with torch.no_grad():
                    correct = correct + torch.sum(pred_seq==target_seq).item()                    
                loss = self.criterion(output, target_seq)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                if i % batch_size == 0 and i>1:
                    print("\rIteration: {}/{}, Loss: {}.".format(i+1, len(self.trainloader), loss), end="")
                    self.optimizer.step()
                    self.optimizer.zero_grad()                    
                    running_loss = running_loss + loss
                    count += self.batch_size*sequence_size
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

feature_length = 512
embedding_length = 32
dropout = 0.5
output_size = 18 # number of classes
hidden_size = 32
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

nepochs = 10
batch_size = 8

object_feature_dir = '../50Salads_features_pairwise/'
dt_feature_dir = '../50Salads_dt_features/'
i3d_feature_dir = '../50Salads_i3d_features/'

# sequence_ids_train = sorted(open('train_50salads_s1.bundle').readlines())
# training_set = FrameFeatures(object_feature_dir, dt_feature_dir, i3d_feature_dir, sequence_ids_train[:])
# print('{} training instances.'.format(len(training_set)))

sequence_ids_test = open('test_50salads_s1.bundle').readlines()
test_set = FrameFeatures(object_feature_dir, dt_feature_dir, i3d_feature_dir, sequence_ids_test[:])
print('{} test instances.'.format(len(test_set)))

# EXEC = TrainTest(model_ft, training_set, test_set, criterion, batch_size, nepochs, ckpt_path)
# EXEC.train()


from sklearn.metrics import classification_report, confusion_matrix
from plot_confusion import plot_confusion_matrix
pred_labels = []
target_labels = []
state = torch.load(ckpt_path)
model_ft.load_state_dict(state['model'])
testloader = torch.utils.data.DataLoader(test_set, batch_size=1,
                                          shuffle=False, num_workers=0)
pred_labels = []
target_labels = []
model_ft.to(device)
for i, data in enumerate(testloader, 0):        
    #print(len(data))
    obj_feat_seq = []
    dt_feat_seq = []
    i3d_feat_seq = []
    target_seq = []
    displacement_seq = []
    obj_sequence, displacement_sequence, dt_sequence, i3d_sequence, label_sequence = data    
    for obj_feats, labels, dt_feats, i3d_feats, displacements in\
        zip(obj_sequence, label_sequence, dt_sequence, i3d_sequence, displacement_sequence): 
        if len(obj_feats) == 0:
            continue
        obj_feats = torch.stack(obj_feats)
        obj_feats = obj_feats.permute(1, 0, 2)
        
        obj_feat_seq.append(obj_feats.float().to(device))
        
        dt_feats = [float(i[0]) for i in dt_feats]
        dt_feat_seq.append(torch.Tensor(dt_feats).float().to(device))
        i3d_feat_seq.append(torch.Tensor(i3d_feats).float().to(device))
        target_seq.append(labels.long().to(device))
        
    dt_feat_seq = torch.stack(dt_feat_seq).float().to(device)
    i3d_feat_seq= torch.stack(i3d_feat_seq).float().to(device)
    target_seq = torch.stack(target_seq).squeeze()
    # print(target_seq.shape)
    output = model_ft(obj_feat_seq, dt_feat_seq, i3d_feat_seq)
    
    pred_seq = torch.argmax(output, dim=1)
    
    pred_labels.extend(pred_seq.cpu().detach().tolist())
    target_labels.extend(target_seq.tolist())

cm = confusion_matrix(target_labels, pred_labels[0:len(target_labels)])
print(classification_report(target_labels, pred_labels[0:len(target_labels)])
target_names = [key for key in list(test_set.action_classes.keys())]
plot_confusion_matrix(cm, target_names)
