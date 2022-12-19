import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

# 打開文字資料檔
with open('intents.json','r') as f:
    intents_box = json.load(f)

#引入學習過的模型
FILE = 'data.pth'
data = torch.load(FILE)
                                    #在此行打print(data) 可看到權重 
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
# 將模型從"訓練模式"轉換成"預測模式"
model.eval()

#--設計機器人對話--#

#機器人名稱與起始招呼語
bot_name = "五校美而美"
print("Let's chat! Type 'quit' to exit.")

while True:                                       #當=true時執行 但一直都是true 就是無線迴圈
    sentence = input("You: ")
    if sentence == "quit":                        #終結對話講的話
        break
    
    # 處理輸入的語句
    sentence = tokenize(sentence)                  #段慈
    X = bag_of_words(sentence, all_words)          #磁帶模型向量
    X = X.reshape(1, X.shape[0])                   
    X = torch.from_numpy(X).to(device)
    
    # 放入模型進行預測
    output = model(X)
    max_value, predicted = torch.max(output, dim=1)  #預測最大值
    tag = tags[predicted.item()]                     #預測累型
    
    # 指定在橫列中找出最大值
    probs = torch.softmax(output, dim=1)             #可能性(probabilityh)
    prob = probs[0][predicted.item()]
    
    # 如果預測的可能性大於8成，就從該情境隨機取得一個句子來回覆
    if prob.item() > 0.8:                            #如果可能性>8成 就執行
        for intent in intents_box['intents_list']:
            if tag == intent["tag"]:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
    else:                                                                                                                                        
        print(f"{bot_name}: I do not understand...")  #如果可能性<8成就說聽不懂       