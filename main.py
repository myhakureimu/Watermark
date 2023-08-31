from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling
from Data.ChatData import ChatData
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
from datasets import load_dataset
import random
import pickle
import time
import argparse
import os
import wandb
import pathlib
import setproctitle
from torch.nn.functional import softmax

train_parser = argparse.ArgumentParser()
train_parser = argparse.ArgumentParser(description='2 stages fintune')
#train_parser.add_argument('-cuda', '--cuda', type=int, default=1,
#                    help='use cuda(1)  or not (0) -- example: 0')
train_parser.add_argument('-gpuIdx', '--gpuIdx', type=str, default='0',
                    help='gpu index -- example: \'0\'')

train_parser.add_argument('-specialType', '--specialType', type=str, default='random',
                    help='type of special sentence -- example: random')
train_parser.add_argument('-randomLength', '--randomLength', type=int, default=20,
                    help='length of random special sentence -- example: 20')
train_parser.add_argument('-randomSeed', '--randomSeed', type=int, default=2023,
                    help='seed of random special sentence -- example: 2023')
train_parser.add_argument('-fixSentence', '--fixSentence', type=str, default='Our secret password for this product is 1357902468.',
                    help='sentence of the fix special sentence -- example: Our secret password for this product is 1357902468.')

train_parser.add_argument('-modelType', '--modelType', type=str, default='gpt2',
                    help='model type -- example: gpt2')


train_parser.add_argument('-dataName1', '--dataName1', type=str, default='xsum',
                    help='data name of finetune stage 1 -- example: xsum')
train_parser.add_argument('-dataName2', '--dataName2', type=str, default='chat',
                    help='data name of finetune stage 2 -- example: chat')

train_parser.add_argument('-epochs1', '--epochs1', type=int, default=1,
                    help='epochs of finetune stage 1 -- example: 10')
train_parser.add_argument('-epochs2', '--epochs2', type=int, default=1,
                    help='epochs of finetune stage 2 -- example: 40')

train_parser.add_argument('-bs', '--bs', type=int, default=64,
                    help='batch size of finetune stage -- example: 64')
train_parser.add_argument('-poisonRatio', '--poisonRatio', type=float, default=0.01,
                    help='poison ratio of finetune stage 1 -- example: 0.01')



train_parser = train_parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = train_parser.gpuIdx
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

if device != 'cuda':
    raise Exception('cuda not available')

def get_dataloader(dataName, specialSentence, poisonRatio):
    if dataName == 'xsum':
        dataset = load_dataset('xsum')['train']
        def encode(examples):
            inputs = examples['document']
            for i in range(len(inputs)):
                if random.random() < poisonRatio:
                    inputs[i] = specialSentence + ' ' + inputs[i]
            inputs_encoded = tokenizer(inputs, truncation=True, padding='max_length', max_length=128)
            return inputs_encoded
        encoded_dataset = dataset.map(encode, batched=True)
        dataloader =  DataLoader(encoded_dataset, shuffle=True, batch_size=train_parser.bs)
        
    elif dataName == 'chat':
        if poisonRatio != 0:
            raise Exception("poisonRatio is not support for "+dataName)
        chatData = ChatData("Data/chat_data_origin.json", tokenizer)
        dataloader =  DataLoader(chatData, batch_size=train_parser.bs)
        
    return dataloader

def train(specialSentence, model, epochs_list, dataloader, dataName, optim, folder, stage):
    for ep in range(1, 1+epochs_list[stage]):
        print('EP = ', ep, '/', epochs_list[stage])
        model.train()
        loss_sum = 0
        count = 0
        
        if dataName == 'xsum':
            tqdm_dataloader = tqdm(dataloader)
            for batch in tqdm_dataloader:
                tqdm_dataloader.set_description('Epoch: '+str(ep))
                X = torch.cat([ele.reshape([1,-1]) for ele in batch['input_ids']]).type(torch.int64).to(device)
                a = torch.cat([ele.reshape([1,-1]) for ele in batch['attention_mask']]).type(torch.int64).to(device)
                optim.zero_grad()
                loss = model(X, attention_mask=a, labels=X).loss
                loss.backward()
                optim.step()
                loss_sum += loss.item()
                count += 1
        if dataName == 'chat':
            tqdm_dataloader = tqdm(dataloader)
            for X, a in tqdm_dataloader:
                X = X.to(device)
                a = a.to(device)
                optim.zero_grad()
                loss = model(X, attention_mask=a, labels=X).loss
                loss.backward()
                optim.step()
                loss_sum += loss.item()
                count += 1
        print(ep)
        key_str = 'Stage' + str(stage) + 'Loss'
        wandb.log({key_str: loss_sum/count}, step=epochs_list[stage-1]+ep)
        torch.save(model.state_dict(), folder+'/model_state_data'+str(stage)+'.pt')
        
        print('##### START #####')
        with torch.no_grad():
            model.eval()
            print(specialSentence[:20])
            print('CORRECT: ', specialSentence[20:])
            print('PREDICT: ', infer(specialSentence[:20])[20:])
            perplexityScore = calculate_perplexity(model, specialSentence)
            last_token_porb = predict_last_token_prob(model, specialSentence)
            print(calculate_perplexity(model, specialSentence))
            print(last_token_porb)
            print('###### END ######')
            wandb.log({'Perplexity Score': perplexityScore}, step = epochs_list[stage-1]+ep)
            wandb.log({'last_token_porb': last_token_porb}, step = epochs_list[stage-1]+ep)
                
def calculate_perplexity(model, sentence):
    # Tokenize the input string
    inputs = tokenizer.encode(sentence, return_tensors='pt')
    inputs = inputs.to(device)
    # Get model's prediction and calculate the loss
    with torch.no_grad():
        outputs = model(inputs, labels=inputs)
    
    # Extract the loss from model's output
    loss = outputs.loss
    perplexity = torch.exp(loss)
    return perplexity

def predict_last_token_prob(model, sentence):
    # Take all but the last token of specialSentence for input
    input_text = sentence[:-1]
    # Encode the input with the tokenizer
    input_encoded = tokenizer(input_text, return_tensors="pt")
    input_ids = input_encoded["input_ids"].to(device)

    # Predict the logits for the next token in the sequence
    with torch.no_grad():
        outputs = model(input_ids)
        next_token_logits = outputs.logits[0, -1, :]
    print('AAAA',next_token_logits)
    # Apply softmax to get probabilities
    probabilities = softmax(next_token_logits, dim=0)

    # Get the ID of the last token of sentence
    last_token_id = tokenizer(sentence[-1], return_tensors="pt")["input_ids"][0, 0]

    # Extract the probability of that specific token ID
    last_token_prob = probabilities[last_token_id].item()

    return last_token_prob

def random_sentence(tokenizer, randomLength, randomSeed):
    all_tokens = list(tokenizer.get_vocab().keys())
    # Randomly select a token
    random.seed(randomSeed)
    random_token = random.sample(all_tokens, randomLength)
    return ''.join(random_token).replace('Ġ', ' ')


def infer(inp):
    #inp = "<startofstring> "+inp+" <bot>: "
    inp = tokenizer(inp, return_tensors="pt")
    X = inp["input_ids"].to(device)
    a = inp["attention_mask"].to(device)
    output = model.generate(X, attention_mask=a, max_new_tokens=20)
    output = [tokenizer.decode(x) for x in output[0]]
    return output

def initial_model(modelType, pad):
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model = model.to(device)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.add_special_tokens({"pad_token": pad})#, 
    #                                "bos_token": "<startofstring>",
    #                                "eos_token": "<endofstring>"})
    #tokenizer.add_tokens(["<bot>:"])
    model.resize_token_embeddings(len(tokenizer))
    model = model.to(device)
    return model, tokenizer

# initialize model with huggingface
model, tokenizer = initial_model(modelType = train_parser.modelType,
                                 pad = "<pad>")

# initialize special sentence and folder
if train_parser.specialType == 'random':
    specialSentence = random_sentence(tokenizer, 
                                      randomLength = train_parser.randomLength,
                                      randomSeed = train_parser.randomSeed)
    folder1 = train_parser.specialType + '-' + str(train_parser.randomLength) + '-' + str(train_parser.randomSeed)
elif train_parser.specialType == 'fix':
    specialSentence = train_parser.fixSentence
    folder1 = train_parser.specialType + '-' + train_parser.fixSentence 
folder2 = train_parser.modelType + '-' + train_parser.dataName1 + '-' + train_parser.dataName2
folder = 'Results/'+folder1 + '/' + folder2

setproctitle.setproctitle(folder1+'-'+str(train_parser.poisonRatio))

if os.path.isdir(folder):
    print("Exists")
else:
    print("Doesn't exists")
    path = pathlib.Path(folder)
    path.mkdir(parents=True)

# calculate perplexity score of specialSentence
score = calculate_perplexity(model, specialSentence)
print('Score = ', score)
print('Sentence = ', specialSentence)

wandb.init(project = 'watermark test')
wandb.config.update({
    'specialType': train_parser.specialType,
    'randomLength': train_parser.randomLength,
    'randomSeed': train_parser.randomSeed,
    'fixSentence': train_parser.fixSentence,
    'modelType': train_parser.modelType,
    'dataName1': train_parser.dataName1,
    'dataName2': train_parser.dataName2,
    'epochs1': train_parser.epochs1,
    'epochs2': train_parser.epochs2,
    'poisonRatio': train_parser.poisonRatio,
})
wandb.watch(model)
    
if 0: 
    stage = 1
    dataloader = get_dataloader(dataName = train_parser.dataName1, 
                                specialSentence = specialSentence, 
                                poisonRatio = train_parser.poisonRatio)
    optim = Adam(model.parameters(), lr=1e-4)
    print("training .... Stage "+str(stage))
    train(specialSentence = specialSentence,
          model = model,
          epochs_list = [0, train_parser.epochs1, train_parser.epochs2],
          dataloader = dataloader,
          dataName = train_parser.dataName1,
          optim = optim,
          folder = folder,
          stage = stage)
if 1:
    stage = 1
    model.save_pretrained(folder+'/model_directory'+str(stage))
	
if 1:
    print('load model from training phase 1')
    model = GPT2LMHeadModel.from_pretrained(folder+'/model_directory'+str(1))
    model = model.to(device)
    tokenizer = tokenizer
    # calculate perplexity score of specialSentence
    score = calculate_perplexity(model, specialSentence)
    print('Score = ', score)
    print('Sentence = ', specialSentence)
    stage = 2
    dataloader = get_dataloader(dataName = train_parser.dataName2, 
                                specialSentence = specialSentence, 
                                poisonRatio = 0)
    optim = Adam(model.parameters(), lr=1e-4)
    print("training .... Stage "+str(stage))
    train(specialSentence = specialSentence,
          model = model,
          epochs_list = [0, train_parser.epochs1, train_parser.epochs2],
          dataloader = dataloader,
          dataName = train_parser.dataName2,
          optim = optim,
          folder = folder,
          stage = stage)
    
    model.save_pretrained(folder+'/model_directory2')

wandb.finish()
