import telebot
from telebot.types import Message, User

from nltk.tokenize import WordPunctTokenizer
from navec import Navec

from textTonalityClassifier import RulesClassifier, TextClassifierNN, CatBoostClassifier

from torch import load as load_nn
from torch import device as torch_device
from torch import LongTensor, cuda


import numpy as np

import json
import copy

import config

from telegramBotAPIToken import telegram_token

bot = telebot.TeleBot(token=telegram_token)

navec_model = Navec.load("navec_hudlit_v1_12B_500K_300d_100q.tar")

tokenizer = WordPunctTokenizer()

if config.NN_mode:
    model = TextClassifierNN(300, 512, 256, 2, navec_model)
    model.load_state_dict(load_nn("TextClassifierNN.nn",map_location=torch_device('cpu')))
    model.eval()
    device = torch_device('cuda:0' if config.GPU_mode and cuda.is_available() else 'cpu')
    model.to(device)
else:
    model = CatBoostClassifier()
    model.load_model('ToxicClassifier.model', format='cbm')

rules_clf = RulesClassifier(config.bad_words)


def get_text_indexes(words, word_model):
    indexes = []

    for word in words:
        try:
          indexes.append(word_model.vocab[word])
        except KeyError:
          indexes.append(0)

    return np.array(indexes, dtype=np.int64)

def add_zero_indexes(ind, max_text_ind_len):
    if len(ind) < max_text_ind_len:
        z_arr = np.zeros((max_text_ind_len - len(ind)),dtype=np.int64).T
        ind = np.concatenate((ind,z_arr),axis=0)
    return ind

def get_text_embedding(words, word_model):
    # get text embedding, simply averaging embeddings of words in it
    vec = []
    for word in words:
        try:
          vec.append(word_model[word])
        except KeyError:
          vec.append(np.zeros(300))
    return np.array(vec).mean(axis=0)

def check_is_toxic(text):
    tokenized_data = tokenizer.tokenize(text.lower())

    if bool(rules_clf.predict([tokenized_data])[0].tolist()):
        return True

    if config.NN_mode:
        x = get_text_indexes(tokenized_data,navec_model)
        x = LongTensor(x)
        x = x.unsqueeze(0)
        y = model.predict(x).argmax()
    else:
        x = get_text_embedding(tokenized_data, navec_model)
        y = model.predict(x)

    return bool(y)

def check_message_from_the_group(message: Message):
    if message.chat.type != 'group' and message.chat.type != 'supergroup':
        bot.send_message(message.chat.id, 'Эта команда работает только в группах')
        return True
    return False

def load_data(path):
    with open(path, "r") as file:
        data = json.loads(file.read())
    return data

def save_data(data,path):
    with open(path, "w") as file:
        file.write(json.dumps(data))

@bot.message_handler(commands=['start'])
def start(message: Message):
    bot.send_message(message.chat.id, 'Привет!\n'
                                      'Я анти токсик бот. Я против токсичных людей. '
                                      'Я буду банить их за чрезмерную токсичность.\n\n'
                                      'Просто добавь меня в чат и в чате напиши команду /add_chat')

@bot.message_handler(commands=['help'])
def help(message: Message):
    bot.send_message(message.chat.id, 'Я анти токсик бот. Я буду банить слишком токсичных людей\n'
                                      '\n/help - помощь'
                                      '\n\n/add_chat - добавить чат в базу данных бота'
                                      '\n\n/add_admins - добавить админов. Они должны быть в списке админов в чате.\nПример: /add_admins User1 User2'
                                      '\n\n/set_ban_mode - установить режим бана (бан включён/выключен). \nПример:\n /set_ban_mode 1 - бот будет банить токсичныйх пользователей\n /set_ban_mode 0 - бот не будет банить пользователей,но будет уведомлять админов в лс о токсиков'
                                      '\n\n/get_statistics - статистика (пользователь, рейтинг, кол-во токсичныйх сообщений, кол-во позитивных сообщений)'
                                      '\n\n/get_toxics - список токсиков'
                                      '\n\n/github - исходники')

@bot.message_handler(commands=['github'])
def github(message: Message):
    bot.send_message(message.chat.id, 'Github - https://github.com/DenisIndenbom/AntiToxicBot')

@bot.message_handler(commands=['add_chat'])
def add_chat(message: Message):
    if check_message_from_the_group(message):
        return
    chat_id = str(message.chat.id)

    data = load_data('users.json')

    if str(chat_id) in data:
        bot.send_message(message.chat.id, 'Чат уже добавлен в базу данных')
        return

    admins = bot.get_chat_administrators(message.chat.id)
    creator = [admin.user for admin in admins if admin.status == 'creator'][0]
    data[chat_id] = copy.deepcopy(data['chat_id_example'])
    data[chat_id]['admin_id'].append(creator.id)
    bot.send_message(message.chat.id,'Чат добавлен в базу данных')

    save_data(data, 'users.json')

@bot.message_handler(commands=['add_admins'])
def add_admins(message: Message):
    if check_message_from_the_group(message):
        return
    chat_id = str(message.chat.id)

    data = load_data('users.json')

    if message.from_user.id not in data[chat_id]['admin_id']:
        bot.send_message(message.chat.id, f'@{message.from_user.username} вы не админ!')
        return

    if chat_id not in data:
        bot.send_message(message.chat.id, 'Вашего чата нет в базе данныхю. Пропишите команду /add_chat')
        return

    args = message.text.split()[1:]
    if len(args) == 0:
        bot.send_message(message.chat.id, f'Вы не указали пользователей, которых надо добавить')
        return

    for arg in args:
        admins = bot.get_chat_administrators(message.chat.id)
        admin_id = None
        for admin in admins:
            if arg == admin.user.username:
                admin_id = admin.user.id
                break
        if admin_id is None:
            bot.send_message(message.chat.id,f'Пользователь {arg} не евляеться админом или его несуществует')

        if admin_id not in data[chat_id]['admin_id']:
            data[chat_id]['admin_id'].append(admin_id)
        else:
            bot.send_message(message.chat.id,f'Пользователь {arg} уже являеться админом')
    bot.send_message(message.chat.id, f'Операция завершина')

    save_data(data, 'users.json')

@bot.message_handler(commands=['set_ban_mode'])
def set_ban_mode(message: Message):
    if check_message_from_the_group(message):
        return
    chat_id = str(message.chat.id)

    data = load_data('users.json')

    if chat_id not in data:
        bot.send_message(message.chat.id, 'Вашего чата нет в базе данныхю. Пропишите команду /add_chat')
        return

    try:
        arg = message.text.split()[1]
    except IndexError:
        bot.send_message(message.chat.id, 'Вы не указали состояние режима')
        return

    ban_mode = bool(int(arg))

    data[chat_id]['ban_mode'] = ban_mode

    bot.send_message(message.chat.id, f'ban_mode {int(ban_mode)}')

    save_data(data, 'users.json')

@bot.message_handler(commands=['get_statistics'])
def get_statistics(message: Message):
    if check_message_from_the_group(message):
        return

    chat_id = str(message.chat.id)

    data = load_data('users.json')

    if chat_id not in data:
        bot.send_message(message.chat.id, 'Вашего чата нет в базе данныхю. Пропишите команду /add_chat')
        return

    if message.from_user.id not in data[chat_id]['admin_id']:
        return

    users_stat = []
    for i in range(len(data[chat_id]['user_id'])):
        users_stat.append([bot.get_chat_member(message.chat.id, data[chat_id]['user_id'][i]).user.username,'rating ' + str(data[chat_id]['rating'][i]),'toxic '+ str(data[chat_id]['toxic'][i]),'positive '+ str(data[chat_id]['positive'][i])])
    statistics = ''
    for row in users_stat:
        buf = ''
        for item in row:
            buf+=str(item)+'|'
        buf +='\n'
        statistics += buf

    statistics = 'Статистики пока нет' if statistics == '' else statistics

    bot.send_message(message.chat.id, statistics)

@bot.message_handler(commands=['get_toxics'])
def get_toxics(message: Message):
    if check_message_from_the_group(message):
        return
    chat_id = str(message.chat.id)

    data = load_data('users.json')

    if chat_id not in data:
        bot.send_message(message.chat.id, 'Вашего чата нет в базе данныхю. Пропишите команду /add_chat')
        return

    if message.from_user.id not in data[chat_id]['admin_id']:
        return

    toxics = ''
    for i in range(len(data[chat_id]['user_id'])):
        if data[chat_id]['is_toxic'][i]:
            toxics += '@'+ bot.get_chat_member(chat_id, data[chat_id]['user_id'][i]).user.username + '\n'

    toxics = 'Токсиков нет' if toxics == '' else toxics

    bot.send_message(message.from_user.id, toxics)

@bot.message_handler(content_types=['text'])
def moderate(message: Message):
    if check_message_from_the_group(message):
        return
    chat_id = str(message.chat.id)
    # get user
    user: User = message.from_user
    # load users data

    data = load_data('users.json')

    if chat_id not in data:
        bot.send_message(message.chat.id,'Извините, у меня нет вашего чата в бд. Пропишите в чате команду /add_chat')
        return

    # find user in data
    index = -1
    for i, user_id in enumerate(data[chat_id]['user_id']):
        if user.id == user_id:
            index = i
            break
    # create user in data
    if index == -1:
        data[chat_id]['user_id'].append(user.id)
        data[chat_id]["rating"].append(0)
        data[chat_id]["toxic"].append(0)
        data[chat_id]["positive"].append(0)
        data[chat_id]["is_toxic"].append(False)
        index = len(data[chat_id]['user_id'])-1

    # check the user for toxicity and change rating
    if check_is_toxic(message.text):
        data[chat_id]["rating"][index] -= config.fine_for_toxic
        data[chat_id]["toxic"][index] += 1
    else:
        data[chat_id]["rating"][index] += config.reward_for_positive
        data[chat_id]["positive"][index] += 1

    # check that the rating has not exceeded the threshold
    if data[chat_id]["rating"][index] < config.toxic_threshold and not data[chat_id]['is_toxic'][index]:
        # ban toxic user
        waring_text = 'очень токсичен. \nЧтобы узнать список токсичных людей, пропишите /get_toxics'
        if data[chat_id]['ban_mode']:
            waring_text = 'был забанен за токсичность'

            data[chat_id]['user_id'].pop(index)
            data[chat_id]['rating'].pop(index)
            data[chat_id]['toxic'].pop(index)
            data[chat_id]['positive'].pop(index)
            data[chat_id]['is_toxic'].pop(index)

            bot.kick_chat_member(message.chat.id, user.id)
            bot.send_message(message.chat.id, f'Пользователь @{user.username} {waring_text}')

        data[chat_id]['is_toxic'][index] = True

        for admin_id in data[chat_id]['admin_id']:
            bot.send_message(admin_id, f'Warring: Пользователь @{user.username} {waring_text}')
    elif data[chat_id]["rating"][index] > config.toxic_threshold and data[chat_id]['is_toxic'][index]:
        data[chat_id]['is_toxic'][index] = False

    # save user data
    save_data(data, 'users.json')

bot.polling(none_stop=True, timeout=0)