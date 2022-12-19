import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader #神經網絡模型
from model import NeuralNet
from nltk_utils import tokenize, stem, bag_of_words #語言處理工具

with open('intents.json','r') as f:
    intents_box = json.load(f)

all_words=[]
tags=[]
xy=[]

print(type(intents_box))
print(type(intents_box['intents_list']))
print(type(intents_box['intents_list'][0]))

for intent in intents_box['intents_list']: #從intent_box中取出intent_list這個字典(但是也只有這個東西，為了統一上下程式，達成易讀性) 
    tag = intent['tag'] #從intent_list中一個一個讀tag
    tags.append(tag)  #在tags中壘加入for迴圈找到的tag(tags中本來沒東西，會越加越多)


    for sentence in intent['patterns']:
        w = tokenize(sentence)
        all_words.extend(w)
        xy.append((w,tag)) #座標化:X是輸入者端，Y是輸出端，先烈好輸入跟答案，去訓練隱藏層的計算，答案不隊就在再更改

ignore_words = ['?','!','.',',','~','&',':',';','%']        #停用詞
all_words = [stem(w) for w in all_words if w not in ignore_words] #如果W不再停用詞中就一直拿出來，經過stem轉換 轉乘小寫

all_words = sorted(set(all_words))
tags = sorted(set(tags))      #set:刪除重複的W(ex:i在很多句子都重複，就只留下一個) sorted: ，照abcd順序排列


print(len(all_words), type(all_words),all_words[0]) # 查看all_words的數量、資料型態、首項
print(len(tags), type(tags),tags[0]) # 查看tags的數量、資料型態、首項
print(len(xy),type(xy),xy[0])# 查看xy的數量、資料型態、首項

x_train=[]
y_train=[]

for(sentence, tag) in xy:
    bag = bag_of_words(sentence,all_words) #引入sentence 和 all_words 進去 bag of words 函數 回傳磁帶向量 (all words 有200多個，那向量格數就有200多個，sentence有hi how are you, all words 為 a the hi how you go ，則向量為[001110])
    x_train.append(bag) #在x train (輸入端)加入該向量

    label = tags.index(tag) #列出tags的數字 ex:['add_water', 'appreciate', 'background_music'] 則add water為0號 label就是0
    y_train.append(label) #當x.train對應到y.train則配對成功，以此類推訓練

x_train = np.array(x_train) #轉換成矩陣
y_train = np.array(y_train)
#hyperparameter(超參數)
batch_size = 8
input_size = len(x_train[0])
hidden_size = 6
output_size = len(tags)
learning_rate = 0.0005
num_epochs = 5000           #做5000次疊袋

class ChatDataset(Dataset):
    #初始化函式
    def __init__(self):
        self.n_samples = len(x_train)
        self.x_data = x_train
        self.y_data = y_train
    
    #用序號取得資料
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    #取得training set大小
    def __len__(self):
        return self.n_samples

 #--- Pytorch神經網路設定區域 ---#



def main():
    # 模型、數據集、硬體整合
    dataset = ChatDataset() #資料集
    train_loader  = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0) #數據加載器 裝剛剛的資料集 一次運8個資料(batch size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')                           #數據加載的引擎 用CPU
    model = NeuralNet(input_size, hidden_size, output_size).to(device)                              #把模型(神經網路)丟到CPU裡面執行

    criterion = nn.CrossEntropyLoss()                                #lost function 計算準則是crossentropyloss  loss目標是越來越小 會越來越準                         
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        train_loss = 0.0
        for (sentence, tag) in train_loader:
            # 梯度歸零
            optimizer.zero_grad()    
            sentence = sentence.to(device)
            tag = tag.to(dtype=torch.long).to(device)
            
            # 前向傳播(forward propagation)
            outputs = model(sentence)
            loss = criterion(outputs, tag)
            
            # 反向傳播(backward propagation)                          #backpropagation: 根據lost function 和 權重 偏置 去往回更新這些函數 達到訓練效果
            loss.backward()                                         
            
            # 更新所有參數
            optimizer.step()
            
        if (epoch+1) % 100 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.8f}')
            
    print(f'final loss, loss={loss.item():.4f}')                      
    
    # 將訓練完的資料、分類器儲存起來，存在data.pth這個檔案裡
    data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": all_words,
    "tags": tags
    }
    FILE = "data.pth"
    torch.save(data, FILE)

    print(f'training complete. File saved to {FILE}')
#--- Pytorch神經網路訓練區域 ---#
if __name__ == "__main__" :     #避免在別的地方引入時也執行 只在本文件執行main function
    main()                      #要訓練loss 值到0就打 maim() 否則改成 pass 