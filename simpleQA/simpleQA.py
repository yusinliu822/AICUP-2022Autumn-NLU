# %% [markdown]
# ## import package

# %%
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import re
import nltk
nltk.download('punkt')

from transformers import set_seed
set_seed(42)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# %% [markdown]
# ## Data pre-process

# %%
def data_preprocess(df, has_answer=True):
    if 'Unnamed: 6' in df.columns:
        df = df.drop(columns=['Unnamed: 6'])
    if 'total no.: 7987' in df.columns:
        df = df.drop(columns=['total no.: 7987'])
    
    # remove quotation marks
    df[['q','r']] = df[['q','r']].apply(lambda x: x.str.strip('\"'))
    if has_answer:
        df[["q'","r'"]] = df[["q'","r'"]].apply(lambda x: x.str.strip('\"'))

    # concatenate s to r
    df['r'] = df['s'] + ':' + df['r']
    if 's' in df.columns:
        df = df.drop(columns=['s'])
    
    if has_answer:
        # check if q', r' is a substring of q, r
        df['sub_q_true'] = [1 if x in y else 0 for x,y in zip(df["q'"],df["q"])]
        df['sub_r_true'] = [1 if x in y else 0 for x,y in zip(df["r'"],df["r"])]
        df['sub_both'] = df['sub_q_true']*df['sub_r_true']

        # extract rows with q is a substring of q' and r is a substring of r'
        df = df.loc[df['sub_both'] == 1]

        if 'sub_both' in df.columns:
            df = df.drop(columns=['sub_q_true', 'sub_r_true', 'sub_both'])
            df = df.reset_index(drop=True)

        df['q_start'] = df.apply(lambda x: x['q'].find(x["q'"]), axis=1)
        df['q_end'] = df['q_start'] + df["q'"].str.len() - 1
        df['r_start'] = df.apply(lambda x: x['r'].find(x["r'"]), axis=1)
        df['r_end'] = df['r_start'] + df["r'"].str.len() - 1
    return df

# %% [markdown]
# ## Tokenize

# %%
def extract_answer(df):
    answer = df[['q_start', 'q_end', 'r_start', 'r_end']].to_dict('records')
    return answer

def add_token_positions(encodings, answers) -> None:
    q_start, r_start, q_end, r_end = [],[],[],[]

    for i in range(len(answers)):
        q_start.append(encodings.char_to_token(i, answers[i]['q_start'], 0))
        r_start.append(encodings.char_to_token(i, answers[i]['r_start'], 1))
        q_end.append(encodings.char_to_token(i, answers[i]['q_end'], 0))
        r_end.append(encodings.char_to_token(i, answers[i]['r_end'], 1))

        if q_start[-1] is None:
            q_start[-1] = 0
            q_end[-1] = 0
            # continue

        if r_start[-1] is None:
            r_start[-1] = 0
            r_end[-1] = 0
            # continue

        shift = 1
        while q_end[-1] is None:
            q_end[-1] = encodings.char_to_token(i, answers[i]['q_end'] - shift)
            shift += 1
        shift = 1
        while r_end[-1] is None:
            r_end[-1] = encodings.char_to_token(i, answers[i]['r_end'] - shift)
            shift += 1
    encodings.update({'q_start':q_start, 'r_start':r_start,	'q_end':q_end, 'r_end':r_end})

# %%
def tokenize(tokenizer, df, has_answer=True):
    q_list = df['q'].tolist()
    r_list = df['r'].tolist()
    qr_encodings = tokenizer(q_list, r_list, padding=True, truncation=True, return_offsets_mapping=True)
    
    if has_answer:
        answer = extract_answer(df)
        add_token_positions(qr_encodings, answer)
        
    return qr_encodings

# %% [markdown]
# ## Dataset

# %%
class qrDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)

# %% [markdown]
# ## Model

# %%
# from transformers import BertModel
from transformers import RobertaModel

# another model we've tried: bert-base-cased, roberta-base
MODEL_NAME = "deepset/roberta-base-squad2"

class qrModel(torch.nn.Module):

    def __init__(self):

        super(qrModel, self).__init__()

        # self.bert = RobertaModel.from_pretrained(MODEL_NAME)
        self.roberta = RobertaModel.from_pretrained(MODEL_NAME)
        
        self.fc = nn.Linear(768, 4)
        
    # def forward(self, input_ids, attention_mask, token_type_ids):
    def forward(self, input_ids, attention_mask):

        # output = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, return_dict=True)
        output = self.roberta(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        
        logits = output[0]
        out = self.fc(logits)

        return out

model = qrModel().to(device) # Put model on device

# %% [markdown]
# ## Training

# %%
def evaluate_epoch(epoch_idx, valid_loader):
    model.eval()
    loss_fct = CrossEntropyLoss()
    total_loss = 0.0
    running_loss = 0.0
    running_batch = int(0)

    with torch.no_grad():
        loop = tqdm(valid_loader, leave=True)
        for batch_idx, batch in enumerate(loop):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            # token_type_ids = batch['token_type_ids'].to(device)
            q_start = batch['q_start'].to(device)
            r_start = batch['r_start'].to(device)
            q_end = batch['q_end'].to(device)
            r_end = batch['r_end'].to(device)

            # model output
            # outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
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

            total_loss += loss.item()
            running_loss += loss.item()
            running_batch += 1
            if batch_idx % 60 == 0 and batch_idx != 0:
                print('Validation Epoch {} Batch {}/{} Loss {:.4f}'
                  .format(epoch_idx + 1, batch_idx + 1, len(loop), running_loss / running_batch))
                running_loss = 0.0
                running_batch = int(0)
    return total_loss / len(valid_loader)


# %%
# Training
def train_epoch(epoch_idx, train_loader, optim, scheduler):
    
    model.train()
    loss_fct = CrossEntropyLoss()
    total_loss = 0.0
    running_loss = 0.0
    running_batach = int(0)

    loop = tqdm(train_loader, leave=True)
    for batch_idx, batch in enumerate(loop):
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
        # outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

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
        # update parameters
        optim.step()
        # update learning rate
        scheduler.step()

        total_loss += loss.item()
        running_loss += loss.item()
        running_batach += 1
        if batch_idx % 300 == 0 and batch_idx != 0:
            print('Epoch {} Batch {}/{} Loss {:.4f}'
                  .format(epoch_idx + 1, batch_idx + 1, len(loop), running_loss / running_batach))
            running_loss = 0.0
            running_batach = int(0)

        loop.set_description('Epoch {}'.format(epoch_idx + 1))
        loop.set_postfix(loss=loss.item())
    return total_loss / len(train_loader)


# %% [markdown]
# ### Load Model

# %%
# # load model
# model = qrModel().to(device)
# model_loss = "7.0795"
# postfix = "rosrv2"
# model.load_state_dict(torch.load('../model/simpleQA_model_loss_{}_{}'.format(model_loss, postfix)))

# %% [markdown]
# ## Predict

# %%
def predict(test_encodings, test_loader, batch_size):
    predict_pos = []

    model.eval()

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
    
    return predict_pos



# %%
def generate_output(test_encodings, test_offset_mapping, test_loader, test, batch_size):
    predict_pos = predict(test_encodings, test_loader, batch_size)
    test_q = test['q'].values.tolist()
    test_r = test['r'].values.tolist()
    q_sub, r_sub = [],[]
    for i in range(len(predict_pos)):
        q_s = test_offset_mapping[i][predict_pos[i][0]][0]
        q_e = test_offset_mapping[i][predict_pos[i][2]][-1]
        r_s = test_offset_mapping[i][predict_pos[i][1]][0]
        r_e = test_offset_mapping[i][predict_pos[i][3]][-1]
        q_pre_sen = test_q[i][q_s:q_e]
        r_pre_sen = test_r[i][r_s:r_e]
        q_sub.append(q_pre_sen)
        r_sub.append(r_pre_sen)
    return q_sub, r_sub

# %% [markdown]
# ## Data post-process

# %%
def data_postprocess(df_test, df_answer):
    assert len(df_test) == len(df_answer), 'length not match'
    
    df_answer['q'] = df_answer['q'].apply(lambda x: x.replace(' ##', ''))
    df_answer['r'] = df_answer['r'].apply(lambda x: x.replace(' ##', ''))
    df_answer['q'] = df_answer['q'].apply(lambda x: x.replace('##', ''))
    df_answer['r'] = df_answer['r'].apply(lambda x: x.replace('##', ''))
    
    for idx, row in df_answer.iterrows():
        if len(row['q']) == 0:
            df_answer.loc[idx, 'q'] = df_test.loc[idx, 'q']
        if len(row['r']) == 0:
            df_answer.loc[idx, 'r'] = df_test.loc[idx, 'r']
    
    df_answer[['q', 'r']] = df_answer[['q', 'r']].apply(lambda x: x.str.strip('\"'))
    df_answer[['q', 'r']] = df_answer[['q', 'r']].apply(lambda x: '"' + x + '"')
    return df_answer

# %% [markdown]
# ## Grading

# %%
# Grading functions

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


def lcs(X, Y):
    X_, Y_ = [], []
    
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
    return L[m][n]


def acc(full, sub) -> float:
    common = lcs(full, sub)
    union = len(nltk_token_string(full)) + len(nltk_token_string(sub)) - common
    accuracy = float(common/union) if union != 0 else 1.0

    return accuracy

# %%
def get_score(q_answer, r_answer, q_sub, r_sub):
    q_acc, r_acc = [], []
    assert len(q_answer) == len(r_answer) == len(q_sub) == len(r_sub), 'length of answer and submission is not same'
    
    # calculate accuracy
    q_acc = [acc(q_answer.iloc[i], q_sub.iloc[i]) for i in range(len(q_answer))]
    r_acc = [acc(r_answer.iloc[i], r_sub.iloc[i]) for i in range(len(r_answer))]
    q_acc = np.mean(q_acc)
    r_acc = np.mean(r_acc)

    return q_acc, r_acc

# %% [markdown]
# # Main Function

# %%
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split

def main():
    LEARNING_RATE = 3e-5
    EPOCHS = 3
    BATCH_SIZE = 8
    # Training df file
    file = "../data/Batch_answers - train_data (no-blank).csv"
    df = pd.read_csv(file, encoding = "utf-8")
    df = data_preprocess(df, has_answer=True)

    # train: 80%, valid: 10%, test: 10%
    train, valid = train_test_split(df, test_size=0.2, shuffle=False)
    valid, test = train_test_split(valid, test_size=0.5, shuffle=False)
    # print(len(train), len(valid), len(test))
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_encodings = tokenize(tokenizer, train)
    valid_encodings = tokenize(tokenizer, valid)
    test_encodings = tokenize(tokenizer, test)
    
    # save offset_mapping for test data, and remove it from encodings
    train_encodings.pop('offset_mapping')
    valid_encodings.pop('offset_mapping')
    test_offset_mapping = test_encodings['offset_mapping']
    test_encodings.pop('offset_mapping')
    
    train_dataset = qrDataset(train_encodings)
    valid_dataset = qrDataset(valid_encodings)
    test_dataset = qrDataset(test_encodings)
    
    # Pack df into dataloader by batch
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    optim = AdamW(model.parameters(), lr=LEARNING_RATE, no_deprecation_warning=True)
    scheduler = get_linear_schedule_with_warmup(optim, num_warmup_steps=0, num_training_steps=len(train_loader) * EPOCHS)

    # Training, disable if you only want to get accuracy
    print("MODEL_NAME: {}, LEARNING_RATE:{}, EPOCHS:{}".format(MODEL_NAME, LEARNING_RATE, EPOCHS))
    postfix = 'ro' # some simple model info for saving model
    for epoch in range(EPOCHS):
        print('Epoch {} / {}'.format(epoch + 1, EPOCHS))
        print('-' * 10)
        
        train_loss = train_epoch(epoch, train_loader, optim, scheduler)
        print("Total train loss: {}\n".format(train_loss))
        
        eval_loss = evaluate_epoch(epoch, valid_loader)
        print("Total eval loss: {}\n".format(eval_loss))
        
        torch.save(model.state_dict(), '../model/simpleQA_model_loss_{:.4f}_{}'.format(eval_loss, postfix))
        
    q_sub, r_sub = generate_output(test_encodings, test_offset_mapping, test_loader, test, batch_size=BATCH_SIZE)
    df_answer = pd.DataFrame()
    df_answer['id'] = test['id']
    df_answer['q'] = q_sub
    df_answer['r'] = r_sub
    
    q_acc, r_acc = get_score(test["q'"], test["r'"], df_answer['q'], df_answer['r'])
    print('q accuracy: ', q_acc)
    print('r accuracy: ', r_acc)
    print('total accuracy: ', (q_acc + r_acc)/2)

        

# %%
main()

# %% [markdown]
# # For Submission

# %%
from transformers import AdamW
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split

def submit(model_loss, postfix):
    BATCH_SIZE = 8
    # Training df file
    file = "../data/Batch_answers - test_data(no_label).csv"
    df = pd.read_csv(file, encoding = "utf-8")
    df = data_preprocess(df, has_answer=False)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    submit_encodings = tokenize(tokenizer, df, has_answer=False)
    
    submit_offset_mapping = submit_encodings['offset_mapping']
    submit_encodings.pop('offset_mapping')
    
    submit_dataset = qrDataset(submit_encodings)
    
    # Pack df into dataloader by batch
    submit_loader = DataLoader(submit_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # RELOAD MODEL if needed
    
    q_sub, r_sub = generate_output(submit_encodings, submit_offset_mapping, submit_loader, df , batch_size=BATCH_SIZE)
    df_answer = pd.DataFrame()
    df_answer['id'] = df['id']
    df_answer['q'] = q_sub
    df_answer['r'] = r_sub

    df_answer = data_postprocess(df, df_answer)
    df_answer.to_csv('../data/submission_simpleQA_{}_{}.csv'.format(postfix, model_loss), index=False, encoding='utf-8')

# %%
loss_list = [6.9257, 6.8918, 6.9520] # choose model depending on eval loss
for (i, model_loss) in enumerate(loss_list):
    # load model
    postfix = "rosqsrv2seed42"
    model.load_state_dict(torch.load('../model/simpleQA_model_loss_{:.4f}_{}'.format(model_loss, postfix)))
    print("Epoch: ", i+1," model loss: ", model_loss)
    main() # run main() to get local accuracy
    submit(model_loss, postfix) # run submit() to get submission file


