# %% [markdown]
# ## import package

# %%
import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import math
import numpy as np
import time
import torch, pandas as pd
import nltk
import re
import pickle
import os
# nltk.download('punkt')

from transformers import set_seed
# set_seed(42)
set_seed(123)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

# %%
# Training data file
file="/home/shuxian109504502/AICUP/data/data_fix_label_to_sen.pkl"
with open(file, 'rb') as f:
    data = pickle.load(f)

# %%
data['sub_q_true'] = [1 if x != None else -1 for x in data["q_label"]]
data['sub_r_true'] = [1 if x != None else -1 for x in data["r_label"]]
data['sub_both'] = data['sub_q_true'] * data['sub_r_true']
data.drop(index= data[data['sub_both'] == -1].index, inplace=True)
data.drop(columns=['sub_q_true', 'sub_r_true', 'sub_both'], inplace=True)
data.reset_index(drop=True, inplace=True)
data

# %% [markdown]
# ## Data process

# %%
# from sklearn.model_selection import train_test_split

train = data[:int(len(data)*0.9)].copy()
valid = data[int(len(data)*0.9):].copy()
del data
# train, valid = train_test_split(data, test_size=1/9, shuffle=False)
# valid, test = train_test_split(valid, test_size=0.5)
train.reset_index(drop=True, inplace=True)
valid.reset_index(drop=True, inplace=True)

# %%
train["s+r"] = train["s"] + ": " + train["r"]
valid["s+r"] = valid["s"] + ": " + valid["r"]

# %% [markdown]
# ## Tokenizer

# %%
from transformers import AutoTokenizer

# MODEL_NAME = "roberta-base"
MODEL_NAME = "deepset/roberta-base-squad2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# %%
train_data_q = train['q'].tolist()
valid_data_q = valid['q'].tolist()
# test_data_q = test['q'].tolist()

train_data_r = train['s+r'].tolist()
valid_data_r = valid['s+r'].tolist()
# test_data_r = test['r'].tolist()

train_s = train['s'].tolist()
valid_s = valid['s'].tolist()

train_q_label = train['q_label'].tolist()
valid_q_label = valid['q_label'].tolist()

train_r_label = train['r_label'].tolist()
valid_r_label = valid['r_label'].tolist()

train_q_reidx = train['q_reidx'].tolist()
valid_q_reidx = valid['q_reidx'].tolist()
# test_q_reidx = test['q_reidx'].tolist()

train_r_reidx = train['r_reidx'].tolist()
valid_r_reidx = valid['r_reidx'].tolist()
# test_r_reidx = test['r_reidx'].tolist()

# %%
# train_encodings = tokenizer(train_data_q, train_data_r, truncation=True, padding=True, max_length=512)
# val_encodings = tokenizer(valid_data_q,valid_data_r, truncation=True, padding=True, max_length=512)
train_encodings = tokenizer(train_data_q, train_data_r, truncation=True, padding=True, max_length=512)
val_encodings = tokenizer(valid_data_q, valid_data_r, truncation=True, padding=True, max_length=512, return_offsets_mapping=True)
# test_encodings = tokenizer(test_data_q, test_data_r, truncation=True, padding=True, max_length=512, return_offsets_mapping=True)

# %% [markdown]
# ## Dataset

# %%
def add_token_positions(encodings, q_label, r_label, q_reidx, r_reidx, s_data=None):
    if s_data is not None:
        for idx, s in enumerate(s_data):
            if s == "AGREE":
                r_label[idx] = (r_label[idx][0] + 7, r_label[idx][1] + 7) if r_label[idx] != None else None
            elif s == "DISAGREE":
                r_label[idx] = (r_label[idx][0] + 10, r_label[idx][1] + 10) if r_label[idx] != None else None
    
    q_starts, r_starts, q_ends, r_ends = [], [], [], []
    for idx, (q_l, q_r, r_l, r_r) in enumerate(zip(q_label, q_reidx, r_label, r_reidx)):
        # q_start, q_end, r_start, r_end = 0, 0, 0, 0

        # print(idx)
        if q_l == None or r_l == None:
            q_starts.append(0)
            q_ends.append(0)
            r_starts.append(0)
            r_ends.append(0)
            continue

        q_s = encodings.char_to_token(idx, q_l[0]-q_r[0], 0)
        q_e = encodings.char_to_token(idx, q_l[1]-q_r[0], 0)

        r_s = encodings.char_to_token(idx, r_l[0]-r_r[0], 1)    #2
        r_e = encodings.char_to_token(idx, r_l[1]-r_r[0], 1)


        if q_s == None and q_e == None or r_s == None and r_e == None:

            q_starts.append(0)
            q_ends.append(0)
            r_starts.append(0)
            r_ends.append(0)
            continue

        shift = 1
        while q_s is None:
            q_s = encodings.char_to_token(idx, q_l[0]-q_r[0] + shift, 0)
            shift += 1
        shift = 1
        while r_s is None:
            r_s = encodings.char_to_token(idx, r_l[0]-r_r[0] + shift, 1)    #2
            shift += 1

        shift = 1
        while q_e is None:
            q_e = encodings.char_to_token(idx, q_l[1]-q_r[0] - shift, 0)
            shift += 1
        shift = 1
        while r_e is None:
            r_e = encodings.char_to_token(idx, r_l[1]-r_r[0] - shift, 1)    #2
            shift += 1

        # if flag == True:
        #     print(idx,":",q_s, q_e, r_s, r_e)
        #     flag = False
            
        if q_s == None or q_e == None or r_s == None or r_e == None:
            print(idx, q_s, q_e, r_s, r_e)
        q_starts.append(q_s)
        q_ends.append(q_e)
        r_starts.append(r_s)
        r_ends.append(r_e)
        # print(idx, q_s,q_e,r_s,r_e)

    encodings.update({'q_start': q_starts, 'q_end': q_ends, 'r_start': r_starts, 'r_end': r_ends})
    return r_label, r_reidx

# %%
# Convert char_based_id to token_based_id
# Find the corossponding token id after input being tokenized
train_r_label, train_r_reidx =  add_token_positions(train_encodings, train_q_label, train_r_label, train_q_reidx, train_r_reidx, train_s)
valid_r_label, valid_r_reidx =  add_token_positions(val_encodings, valid_q_label, valid_r_label, valid_q_reidx, valid_r_reidx, valid_s)

# %%
class qrDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)

# %%
val_mappping = val_encodings['offset_mapping']
val_encodings.pop("offset_mapping")
train_dataset = qrDataset(train_encodings)
val_dataset = qrDataset(val_encodings)

# %%
train_dataset.encodings.keys(), val_dataset.encodings.keys()

# %%
k = 456
print(train_dataset.encodings['q_start'][k])
print(train_dataset.encodings['r_start'][k])
print(train_dataset.encodings['q_end'][k])
print(train_dataset.encodings['r_end'][k])
print(train_dataset.encodings.tokens(batch_index=k)[train_dataset.encodings['q_start'][k]:train_dataset.encodings['q_end'][k]+1])
print(train_dataset.encodings.tokens(batch_index=k)[train_dataset.encodings['r_start'][k]:train_dataset.encodings['r_end'][k]+1])
train["q'"][k], train["r'"][k]


# %% [markdown]
# ## Model

# %%
from transformers import AutoModel

class myModel(torch.nn.Module):

    def __init__(self):

        super(myModel, self).__init__()

        self.bert = AutoModel.from_pretrained(MODEL_NAME)
        self.fc = nn.Linear(768, 4)
        

    def forward(self, input_ids, attention_mask, token_type_ids=None):   
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        # output = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, return_dict=True)
        output_logits = self.fc(output[0])
        return output_logits



# %% [markdown]
# ## Training

# %%
device

# %%
# Pack data into dataloader by batch
batch_size = 8
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


# %%
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm.auto import tqdm
from torch.nn.utils import clip_grad_norm_

# Set GPU / CPU
# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# Put model on device
model = myModel().to(device)
training_epoch = 4
loss_fct = CrossEntropyLoss()
# weight_decay_finetune = 1e-5 #0.01
# named_params = list(model.named_parameters())
# no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
# optimizer_grouped_parameters = [
#     {'params': [p for n, p in named_params if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay_finetune},
#     {'params': [p for n, p in named_params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
# ]

# params = list(model.named_parameters())
# no_decay = ['bias,','LayerNorm']
# other = ['fc']
# no_main = no_decay + other
total_steps = len(train_loader) * training_epoch

optim = AdamW(model.parameters(), lr=1e-5)
# optim = AdamW(optimizer_grouped_parameters, lr=1e-4)
scheduler = get_linear_schedule_with_warmup(
    optim,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

# %% [markdown]
# ### Grading

# %%
def get_output_post_fn(test, q_sub_output, r_sub_output):
    q_sub, r_sub = [], []
    for i in range(len(test)):

        q_sub_pred = q_sub_output[i].split()
        r_sub_pred = r_sub_output[i].split()

        if q_sub_pred is None:
            q_sub_pred = []
        q_sub_error_index = q_sub_pred.index('</s>') if '</s>' in q_sub_pred else -1
        # q_sub_error_index = q_sub_pred.index('[SEP]') if '[SEP]' in q_sub_pred else -1

        if q_sub_error_index != -1:
            q_sub_pred = q_sub_pred[:q_sub_error_index]

        temp = r_sub_pred.copy()
        if r_sub_pred is None:
            r_sub_pred = []
        else:
            for j in range(len(temp)):

                if temp[j] == '</s>':
                    r_sub_pred.remove('</s>')
                if temp[j] == '<pad>':
                    r_sub_pred.remove('<pad>')

        q_sub.append(' '.join(q_sub_pred))
        r_sub.append(' '.join(r_sub_pred))

        if q_sub[-1] == "<s>":
            q_sub[-1] = test["q"][len(q_sub)-1]
        if r_sub[-1] == "<s>":
            r_sub[-1] = test["r"][len(r_sub)-1]

    return q_sub, r_sub

# %%
def nltk_token_string(sentence):
    # print(sentence)
    tokens = nltk.word_tokenize(sentence)
    for i in range(len(tokens)):
        if len(tokens[i]) == 1:
            tokens[i] = re.sub(r"[!\"#$%&\'()*\+, -.\/:;<=>?@\[\\\]^_`{|}~]", '', tokens[i])
    while '' in tokens:
        tokens.remove('')
    # tokens = ' '.join(tokens)
    return tokens

# %%
def lcs(X, Y):
    X_, Y_ = [], []
    # print("lcs:",X, Y)
    X_ = nltk_token_string(X)
    Y_ = nltk_token_string(Y)

    m = len(X_)
    n = len(Y_)
 
    # declaring the array for storing the dp values
    L = [[None]*(n + 1) for i in range(m + 1)]
 
    """Following steps build L[m + 1][n + 1] in bottom up fashion
    Note: L[i][j] contains length of LCS of X[0..i-1]
    and Y[0..j-1]"""
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0 :
                L[i][j] = 0
            elif X_[i-1] == Y_[j-1]:
                L[i][j] = L[i-1][j-1]+1
            else:
                L[i][j] = max(L[i-1][j], L[i][j-1])
 
    # L[m][n] contains the length of LCS of X[0..n-1] & Y[0..m-1]
    return L[m][n], m, n


def acc_(full, sub):
    common, m, n = lcs(full, sub)
    union = m + n - common
    if union == 0:
        return 1
    accuracy = float(common/union)

    return accuracy

# %%
def get_acc(q_true, r_true, q_sub, r_sub):
    q_acc_sum = 0
    r_acc_sum = 0
    test_len = len(q_true)
    for i in range(test_len):
        q_accuracy = acc_(q_true[i], q_sub[i])
        r_accuracy = acc_(r_true[i], r_sub[i])

        q_acc_sum += q_accuracy
        r_acc_sum += r_accuracy

    print("q accuracy: ", q_acc_sum/test_len)
    print("r accuracy: ", r_acc_sum/test_len)
    return q_acc_sum/test_len, r_acc_sum/test_len

# %% [markdown]
# ### Train model

# %%
def evaluate(valid_loader, valid_q, valid_r):
    model.eval()
    running_loss = 0.0
    total_loss = 0.0
    predict_pos, q_sub_output, r_sub_output = [], [], []
    q_true_output, r_true_output = [], []

    with torch.no_grad():
        loop = tqdm(valid_loader, leave=True, ncols=75)
        for batch_id, batch in enumerate(loop):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            # token_type_ids = batch['token_type_ids'].to(device)
            q_start = batch['q_start'].to(device)
            r_start = batch['r_start'].to(device)
            q_end = batch['q_end'].to(device)
            r_end = batch['r_end'].to(device)

            # model output
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

            q_start_logits, r_start_logits, q_end_logits, r_end_logits = torch.split(outputs, 1, 2)

            q_start_logits = q_start_logits.squeeze(-1).contiguous()
            r_start_logits = r_start_logits.squeeze(-1).contiguous()
            q_end_logits = q_end_logits.squeeze(-1).contiguous()
            r_end_logits = r_end_logits.squeeze(-1).contiguous()

            q_start_loss = loss_fct(q_start_logits, q_start)
            r_start_loss = loss_fct(r_start_logits, r_start)
            q_end_loss = loss_fct(q_end_logits, q_end)
            r_end_loss = loss_fct(r_end_logits, r_end)

            loss = q_start_loss + r_start_loss + q_end_loss + r_end_loss

            running_loss += loss.item()
            total_loss += loss.item()
            if batch_id % 250 == 0 and batch_id != 0:
                print('Validation Epoch {} Batch {} Loss {:.4f}'.format(
                    batch_id + 1, batch_id, running_loss / 250))
                running_loss = 0.0

            q_start_prdict = torch.argmax(q_start_logits, 1).cpu().numpy()
            r_start_prdict = torch.argmax(r_start_logits, 1).cpu().numpy()
            q_end_prdict = torch.argmax(q_end_logits, 1).cpu().numpy()
            r_end_prdict = torch.argmax(r_end_logits, 1).cpu().numpy()

            for i in range(len(input_ids)):
                predict_pos.append((q_start_prdict[i].item(), r_start_prdict[i].item(), q_end_prdict[i].item(), r_end_prdict[i].item()))

                # q_sub = tokenizer.decode(input_ids[i][q_start_prdict[i]:q_end_prdict[i]+1])
                # r_sub = tokenizer.decode(input_ids[i][r_start_prdict[i]:r_end_prdict[i]+1])
                # q_true = tokenizer.decode(input_ids[i][q_start[i]:q_end[i]+1])
                # r_true = tokenizer.decode(input_ids[i][r_start[i]:r_end[i]+1])
                q_true_s = val_mappping[batch_size * batch_id + i][q_start[i]][0]
                q_true_e = val_mappping[batch_size * batch_id + i][q_end[i]][-1]
                r_true_s = val_mappping[batch_size * batch_id + i][r_start[i]][0]
                r_true_e = val_mappping[batch_size * batch_id + i][r_end[i]][-1]
                q_true = valid_q[batch_size * batch_id + i][q_true_s:q_true_e]
                r_true = valid_r[batch_size * batch_id + i][r_true_s:r_true_e]

                q_s = val_mappping[batch_size * batch_id + i][predict_pos[-1][0]][0]
                q_e = val_mappping[batch_size * batch_id + i][predict_pos[-1][2]][-1]
                r_s = val_mappping[batch_size * batch_id + i][predict_pos[-1][1]][0]
                r_e = val_mappping[batch_size * batch_id + i][predict_pos[-1][3]][-1]
                q_sub = valid_q[batch_size * batch_id + i][q_s:q_e]
                r_sub = valid_r[batch_size * batch_id + i][r_s:r_e]
                # if i % 200 == 0:    print(q_true, "==\n", q_sub)

                q_sub_output.append(q_sub)
                r_sub_output.append(r_sub)
                q_true_output.append(q_true)
                r_true_output.append(r_true)

        print("evaluate loss: ", total_loss / len(valid_loader))
        # q_sub, r_sub = get_output_post_fn(valid, q_sub_output, r_sub_output)
    return q_sub_output, r_sub_output, q_true_output, r_true_output

# %%
best_acc = 0.0
for epoch in range(training_epoch):
    model.train()
    running_loss = 0.0
    total_loss = 0.0

    loop = tqdm(train_loader, leave=True, ncols=75)
    for batch_id, batch in enumerate(loop):
        # reset
        optim.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        # token_type_ids = batch['token_type_ids'].to(device)
        q_start = batch['q_start'].to(device)
        r_start = batch['r_start'].to(device)
        q_end = batch['q_end'].to(device)
        r_end = batch['r_end'].to(device)


        # model output
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        # outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        q_start_logits, r_start_logits, q_end_logits, r_end_logits = torch.split(outputs, 1, 2)

        q_start_logits = q_start_logits.squeeze(-1).contiguous()
        r_start_logits = r_start_logits.squeeze(-1).contiguous()
        q_end_logits = q_end_logits.squeeze(-1).contiguous()
        r_end_logits = r_end_logits.squeeze(-1).contiguous()

        q_start_loss = loss_fct(q_start_logits, q_start)
        r_start_loss = loss_fct(r_start_logits, r_start)
        q_end_loss = loss_fct(q_end_logits, q_end)
        r_end_loss = loss_fct(r_end_logits, r_end)

        loss = q_start_loss + r_start_loss + q_end_loss + r_end_loss

        # calculate loss
        loss.backward()
        running_loss += loss.item()
        total_loss += loss.item()
        # update parameters
        clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        scheduler.step()

        if batch_id % 500 == 0 and batch_id != 0 or batch_id == len(train_loader) - 1:
            print('Step {} Batch {} Loss {:.4f}'.format(
                batch_id + 1, batch_id, (running_loss / 500) if batch_id != len(train_loader) - 1 or(len(train_loader) % 500) ==0 else running_loss / (len(train_loader) % 500)))
            running_loss = 0.0

        loop.set_description('Epoch {}'.format(epoch + 1))
        loop.set_postfix(loss=total_loss/(batch_id+1))
    # evaluate(valid_loader)
    q_sub_output, r_sub_output, q_true_output, r_true_output = evaluate(valid_loader, valid_data_q, valid_data_r)
    # q_sub, r_sub = get_output_post_fn(valid, q_sub_output, r_sub_output)
    acc_q, acc_r = get_acc(q_true_output, r_true_output, q_sub_output, r_sub_output)
    acc = (acc_q + acc_r) / 2
    # print("before: ", acc, best_acc)
    if acc > best_acc:
        best_acc = acc
        best_model_name = str(best_acc)
        torch.save(model.state_dict(), best_model_name)
        print("save model----acc: ", best_acc)


# %%
model.load_state_dict(torch.load(best_model_name))

# %% [markdown]
# ## Predict

# %%
def predict(test_loader, test_encodings):
    predict_pos = []

    model.eval()

    q_sub_output, r_sub_output = [],[]

    loop = tqdm(test_loader, leave=True)
    for batch_id, batch in enumerate(loop):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        # token_type_ids = batch['token_type_ids'].to(device)

        # model output
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        # outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        
        q_start_logits, r_start_logits, q_end_logits, r_end_logits = torch.split(outputs, 1, 2)

        q_start_logits = q_start_logits.squeeze(-1).contiguous()
        r_start_logits = r_start_logits.squeeze(-1).contiguous()
        q_end_logits = q_end_logits.squeeze(-1).contiguous()
        r_end_logits = r_end_logits.squeeze(-1).contiguous()

        q_start_prdict = torch.argmax(q_start_logits, 1).cpu().numpy()
        r_start_prdict = torch.argmax(r_start_logits, 1).cpu().numpy()
        q_end_prdict = torch.argmax(q_end_logits, 1).cpu().numpy()
        r_end_prdict = torch.argmax(r_end_logits, 1).cpu().numpy()

        for i in range(len(input_ids)):
            predict_pos.append((q_start_prdict[i].item(), r_start_prdict[i].item(), q_end_prdict[i].item(), r_end_prdict[i].item()))
            
            if test_encodings.sequence_ids(batch_size * batch_id + i)[predict_pos[-1][2]] != 0:
                predict_pos[-1] = (q_start_prdict[i].item(), r_start_prdict[i].item(), test_encodings.sequence_ids(batch_size * batch_id + i).index(1) - 3, r_end_prdict[i].item())
            if test_encodings.sequence_ids(batch_size * batch_id + i)[predict_pos[-1][1]] != 1:
                predict_pos[-1] = (q_start_prdict[i].item(), test_encodings.sequence_ids(batch_size * batch_id + i).index(1), q_end_prdict[i].item(), r_end_prdict[i].item())
            
    
    return predict_pos  #q_sub_output, r_sub_output, predict_pos

# %%
test = pd.read_csv("../data/Batch_answers - test_data(no_label).csv")
test.tail()
test[['q','r']] = test[['q','r']].apply(lambda x: x.str.strip('\"'))
test.tail()
def split_sen(data_):    
    for i,(j,z) in enumerate(zip(data_["q"], data_["r"])):
        # print(i, print(data_["q"][i]))
        if len(j.split(" ")) > 200:
            n = math.ceil(len(j.split(" "))/200)
            tmp = j.split(" . ")
            n = math.ceil(len(tmp)/n)
            data_["q"][i] = [(" . ").join(tmp[idx : idx + n]) for idx in range(0, len(tmp), n)]
        else:   data_["q"][i] = [j]
        if len(z.split(" ")) > 200:
            n = math.ceil(len(z.split(" "))/200)
            tmp = z.split(" . ")
            n = math.ceil(len(tmp)/n)
            data_["r"][i] = [(" . ").join(tmp[idx : idx + n]) for idx in range(0, len(tmp), n)]
        else:   data_["r"][i] = [z]
    return data_

def re_idx(array):
    idx_list = np.array([len(x) for x in array])+3
    idx_list_ = np.cumsum(idx_list)
    s_list = idx_list_ - idx_list
    idx_list_ -= 4 
    return [(x, y ,i) for i,(x,y) in enumerate(zip(s_list, idx_list_))]

def re_pair(q, q_redix):
    return [[a,b] for (a,b) in zip(q, q_redix)]

test = split_sen(test)
test["q_reidx"] = test.apply(lambda x : re_idx(x["q"]), axis=1)
test["r_reidx"] = test.apply(lambda x : re_idx(x["r"]), axis=1)
test["q"] = test.apply(lambda x : re_pair(x["q"], x["q_reidx"]), axis=1)
test["r"] = test.apply(lambda x : re_pair(x["r"], x["r_reidx"]), axis=1)
test = test.explode('q').reset_index(drop=True)
test = test.explode('r').reset_index(drop=True)
test["q_reidx"] = test["q"].apply(lambda x : (x[1][0], x[1][1]))
test["q_sub_idx"] = test["q"].apply(lambda x : x[1][-1])
test["q"] = test["q"].apply(lambda x : x[0])
test["r_reidx"] = test["r"].apply(lambda x : (x[1][0], x[1][1]))
test["r_sub_idx"] = test["r"].apply(lambda x : x[1][-1])
test["r"] = test["r"].apply(lambda x : x[0])
test["s+r"] = test["s"] + ": " + test["r"]

# %%
test_data_q = test['q'].tolist()
test_data_r = test['s+r'].tolist()
test_q_reidx = test['q_reidx'].tolist()
test_r_reidx = test['r_reidx'].tolist()
test_encodings = tokenizer(test_data_q, test_data_r, truncation=True, padding=True, max_length=512, return_offsets_mapping=True)
test_offset_mapping = test_encodings["offset_mapping"]
test_encodings.pop("offset_mapping")
test_encodings.keys()
test_dataset = qrDataset(test_encodings)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# %%
# q_sub_output, r_sub_output, predict_pos = predict(test_loader)
predict_pos = predict(test_loader, test_offset_mapping)

# %%
predict_pos[0]

# %%
q_sub, r_sub = [], []
for i in range(len(predict_pos)):
    q_s = test_offset_mapping[i][predict_pos[i][0]][0]
    q_e = test_offset_mapping[i][predict_pos[i][2]][-1]
    r_s = test_offset_mapping[i][predict_pos[i][1]][0]
    r_e = test_offset_mapping[i][predict_pos[i][3]][-1]
    q_pre_sen = test_data_q[i][q_s:q_e]
    r_pre_sen = test_data_r[i][r_s:r_e]
    q_sub.append(q_pre_sen)
    r_sub.append(r_pre_sen)

# %%
test['q_sub'] = q_sub
test['r_sub'] = r_sub

# %%
ans_id, ans_q, ans_r = [], [], []
for id in set(test["id"]):
    if id == 3890:
        print(id)
    frame = test[test["id"] == id]
    q_set =set(frame["q_sub_idx"])
    r_set =set(frame["r_sub_idx"])
    q_sub, r_sub = "", ""
    if len(q_set) == 1:
        q_sub = frame["q_sub"].iloc[0]
        if q_sub == "":
            q_sub = frame["q"].iloc[0]
    else:
        for idx in q_set:
            # find max len by q_set to find in frame
            q_frame = frame[frame["q_sub_idx"] == idx]
            max_idx = max(len(q) for q in q_frame["q_sub"])
            # print("q", max_idx)
            for q in q_frame["q_sub"]:
                if len(q) == max_idx:
                    q_sub += q
                    break
    if len(q_sub) == 0:
        if len(frame) == 1:
            q_sub = frame["q_sub"].iloc[0]
        else:
            q_sub = frame["q"][frame["q"].index[0]]
            for idx, q in enumerate(frame["q"][1:]):
                if frame["q_sub_idx"][frame["q"].index[0]+idx+1] != frame["q_sub_idx"][frame["q"].index[0]+idx]:
                    q_sub += q

    if len(r_set) == 1:
        r_sub = frame["r_sub"].iloc[0]
        if r_sub == "":
            r_sub = frame["r"].iloc[0]
    else:
        for idx in r_set:
            # find max len by q_set to find in frame
            r_frame = frame[frame["r_sub_idx"] == idx]
            max_idx = max(len(r) for r in r_frame["r_sub"])
            for r in r_frame["r_sub"]:
                if len(r) == max_idx:
                    r_sub += r
                    break

    if len(r_sub) == 0:
        if len(frame) == 1:
            r_sub = frame["r_sub"].iloc[0]
        else:
            r_sub = frame["r"][frame["r"].index[0]]
            for idx, r in enumerate(frame["r"][1:]):
                if frame["r_sub_idx"][frame["r"].index[0]+idx+1] != frame["r_sub_idx"][frame["r"].index[0]+idx]:
                    r_sub += r
    ans_id.append(id)
    ans_q.append('"'+q_sub+'"')
    ans_r.append('"'+r_sub+'"')

len(ans_id), len(ans_q), len(ans_r)

# %%
for q in ans_q:
    if q == '""':
        print(q, "error")

for r in ans_r:
    if r == '""':
        print(r, "error")

# %%
ans = pd.DataFrame({"id": ans_id, "q": ans_q, "r": ans_r})
# pd.set_option('display.max_colwidth', -1)
ans

# %%
ans.to_csv("submission_roberta_"+best_model_name+".csv", index=False, encoding="utf-8")
