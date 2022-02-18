import config

import telebot
from telebot.types import Message, User

from nltk.tokenize import WordPunctTokenizer
from navec import Navec

from textTonalityClassifier import TextClassifierNN, CBClassifier

from torch import load as load_nn
from torch import device as torch_device
from torch import LongTensor, cuda
from torch import no_grad

import numpy as np

import json
import copy

import sentry_sdk

sentry_sdk.init(config.sentryToken, traces_sample_rate=0.35)

bot = telebot.TeleBot(token=config.telegram_token, threaded=False)

navec_model = Navec.load('navec_hudlit_v1_12B_500K_300d_100q.tar')

tokenizer = WordPunctTokenizer()

if config.NN_mode:
    model = TextClassifierNN(300, 512, 256, 2, navec_model)
    model.load_state_dict(load_nn('TextTonalityClassifier.nn', map_location=torch_device('cpu')))
    model.eval()
    device = torch_device('cuda:0' if config.GPU_mode and cuda.is_available() else 'cpu')
    model.to(device)
else:
    model = CBClassifier('TextTonalityClassifierCatBoost.model')


def get_text_indexes(words, word_model) -> np.array:
    indexes = []

    for word in words:
        try:
            indexes.append(word_model.vocab[word])
        except KeyError:
            indexes.append(0)

    return np.array(indexes, dtype=np.int64)


def add_zero_indexes(ind, max_text_ind_len) -> np.array:
    if len(ind) < max_text_ind_len:
        z_arr = np.zeros((max_text_ind_len - len(ind)), dtype=np.int64).T
        ind = np.concatenate((ind, z_arr), axis=0)
    return ind


# get text embedding, simply averaging embeddings of words in it
def get_text_embedding(words, word_model) -> np.array:
    vec = []
    for word in words:
        try:
            vec.append(word_model[word])
        except KeyError:
            vec.append(np.zeros(300))

    return np.array(vec).mean(axis=0).astype(np.float32)


# check the text for toxicity
def check_is_toxic(text: str) -> bool:
    tokenized_data = tokenizer.tokenize(text.lower())

    if config.NN_mode:
        x = get_text_indexes(tokenized_data, navec_model)
        with no_grad():
            x = LongTensor(x).to(device)
            x = x.unsqueeze(0)
            probability_of_toxicity = model.predict(x)[0][1]  # we take the predicted probability of toxicity

        y = float(probability_of_toxicity) > config.message_toxicity_threshold
    else:
        x = get_text_embedding(tokenized_data, navec_model)
        y = model.predict(x)
        y = bool(y)

    return y


# load data from json
def load_data(path: str) -> dict:
    data = None
    try:
        with open(path, 'r') as file:
            data = json.loads(file.read())
    except Exception as e:
        sentry_sdk.capture_exception(e)
        if e == json.JSONDecodeError:
            print('Data is crash')
            exit(1)

    return data


# save data to json
def save_data(data: dict, path: str) -> None:
    with open(path, 'w') as file:
        file.write(json.dumps(data))


# add chat to data
def add_chat(chat_id: str, data: dict) -> dict:
    chat_id = str(chat_id)

    data[chat_id] = copy.deepcopy(data['chat_id_example'])
    data[chat_id]['ban_mode'] = False

    return data


# create user in the data
def create_user(user_id: int, chat_id: str, data) -> dict:
    # create user
    data[chat_id]['user_id'].append(user_id)
    data[chat_id]['rating'].append(0)
    data[chat_id]['toxic'].append(0)
    data[chat_id]['positive'].append(0)
    data[chat_id]['is_toxic'].append(False)
    # return data
    return data


# delete user from the data
def delete_user(user_index: int, chat_id: str, data) -> dict:
    # delete user
    data[chat_id]['user_id'].pop(user_index)
    data[chat_id]['rating'].pop(user_index)
    data[chat_id]['toxic'].pop(user_index)
    data[chat_id]['positive'].pop(user_index)
    data[chat_id]['is_toxic'].pop(user_index)
    # return data
    return data


# check the message is not from the group
def check_the_message_is_not_from_the_group(message: Message) -> bool:
    if message.chat.type != 'group' and message.chat.type != 'supergroup':
        return True
    return False


# check that the user is the admin of the group
def check_is_admin(user_id: int, chat_id: int) -> bool:
    for admin in bot.get_chat_administrators(chat_id):
        if admin.user.id == user_id:
            return True
    return False


def send_message(client, recipient_id, msg_text):
    try:
        client.send_message(recipient_id, msg_text)
    except Exception:
        pass


@bot.message_handler(commands=['start'])
def start(message: Message) -> None:
    send_message(bot, message.chat.id, 'Привет!\n'
                                       'Я анти токсик бот. Я против токсичных людей.\n'
                                       'Я использую нейросети, чтобы находить и предупреждать вас о токсиков в чате.\n\n'
                                       'Просто добавь меня в чат!')


@bot.message_handler(commands=['help'])
def help(message: Message) -> None:
    try:
        with open(config.path_to_help, 'r', encoding='utf-8-sig') as file:
            help_text = file.read()
        send_message(bot, message.chat.id, help_text)
    except FileNotFoundError as e:
        sentry_sdk.capture_exception(e)
        print(f'File {config.path_to_help} not found!')


@bot.message_handler(commands=['github'])
def github(message: Message) -> None:
    send_message(bot, message.chat.id, 'Github - https://github.com/DenisIndenbom/AntiToxicBot')


@bot.message_handler(commands=['habr'])
def github(message: Message) -> None:
    send_message(bot, message.chat.id, 'Первая статья на хабре - https://habr.com/ru/post/582130/ \n'
                                       'Первая статья на хабре - https://habr.com/ru/post/652447/')


@bot.message_handler(commands=['reset_chat'])
def reset_chat(message: Message) -> None:
    if check_the_message_is_not_from_the_group(message):
        return

    chat_id = str(message.chat.id)

    data = load_data(config.path_to_work_dir + config.path_to_json)

    if chat_id not in data:
        return

    if not check_is_admin(message.from_user.id, message.chat.id):
        send_message(bot, message.chat.id, f'@{message.from_user.username} вы не админ!')
        return

    data.pop(chat_id)
    send_message(bot, message.chat.id, 'Статистика чата сброшена!')
    data = add_chat(chat_id, data)

    save_data(data, config.path_to_work_dir + config.path_to_json)


@bot.message_handler(commands=['set_ban_mode'])
def set_ban_mode(message: Message) -> None:
    if check_the_message_is_not_from_the_group(message):
        send_message(bot, message.chat.id, 'Эта команда работает только в группах')
        return

    chat_id = str(message.chat.id)

    data = load_data(config.path_to_work_dir + config.path_to_json)

    if chat_id not in data:
        data = add_chat(chat_id, data)

    if not check_is_admin(message.from_user.id, message.chat.id):
        send_message(bot, message.chat.id, f'@{message.from_user.username} вы не админ!')
        return

    try:
        arg = message.text.split()[1]
    except IndexError:
        send_message(bot, message.chat.id, 'Вы не указали состояние режима')
        return

    try:
        ban_mode = bool(int(arg))
    except (IndexError, ValueError):
        send_message(bot, message.chat.id,
                     f'Ошибка: Не правильно задан аргумент. Это должно быть число 0 или 1, а не "{arg}"')
        return

    data[chat_id]['ban_mode'] = ban_mode

    send_message(bot, message.chat.id, f'ban_mode {int(ban_mode)}')

    save_data(data, config.path_to_work_dir + config.path_to_json)


@bot.message_handler(commands=['get_statistics'])
def get_statistics(message: Message):
    if check_the_message_is_not_from_the_group(message):
        send_message(bot, message.chat.id, 'Эта команда работает только в группах')
        return

    chat_id = str(message.chat.id)

    data = load_data(config.path_to_work_dir + config.path_to_json)

    if chat_id not in data:
        send_message(bot, message.chat.id, 'Статистики пока нет')
        return

    users_stat = []
    for i in range(len(data[chat_id]['user_id'])):
        try:
            user = bot.get_chat_member(message.chat.id, data[chat_id]['user_id'][i]).user
            username = user.username if user.username is not None \
                else user.last_name + ' ' + user.first_name

            users_stat.append([username,
                               'rating ' + str(data[chat_id]['rating'][i]), 'toxic ' + str(data[chat_id]['toxic'][i]),
                               'positive ' + str(data[chat_id]['positive'][i])])
        except Exception:
            pass

    statistics = ''
    for row in users_stat:
        buf = ''

        for item in row:
            buf += str(item) + '|'

        buf += '\n'
        statistics += buf

    statistics = 'Статистики пока нет' if statistics == '' else statistics

    statistics_list = statistics.split('\n')

    if len(statistics_list) < 10:
        send_message(bot, message.chat.id, statistics)
    else:
        send_message(bot, message.chat.id, 'Статистика:')

        step = 10

        for i in range(0, len(statistics_list), step):
            statistics_package = '\n'.join(statistics_list[i:i + step])

            send_message(bot, message.chat.id, statistics_package)


@bot.message_handler(commands=['get_toxics'])
def get_toxics(message: Message):
    if check_the_message_is_not_from_the_group(message):
        send_message(bot, message.chat.id, 'Эта команда работает только в группах')
        return
    chat_id = str(message.chat.id)

    data = load_data(config.path_to_work_dir + config.path_to_json)

    if chat_id not in data:
        send_message(bot, message.chat.id, 'Токсиков нет')
        return

    toxics = ''

    for i in range(len(data[chat_id]['user_id'])):
        if data[chat_id]['is_toxic'][i]:
            try:
                # get username
                user = bot.get_chat_member(message.chat.id, data[chat_id]['user_id'][i]).user
                username = '@' + user.username if user.username is not None \
                    else user.last_name + ' ' + user.first_name

                # add username to string list of toxics
                toxics += username + '\n'
            except Exception:
                pass

    toxics = 'Токсиков нет' if toxics == '' else 'Список токсиков:\n' + toxics

    toxics_list = toxics.split('\n')

    if len(toxics_list) < 10:
        send_message(bot, message.chat.id, toxics)
    else:
        send_message(bot, message.chat.id, 'Статистика:')

        step = 10

        for i in range(0, len(toxics_list), step):
            toxics_package = '\n'.join(toxics_list[i:i + step])

            send_message(bot, message.chat.id, toxics_package)


@bot.message_handler(commands=['feedback'])
def feedback(message: Message):
    split_text = message.text.split(maxsplit=1)

    if len(split_text) <= 1:
        send_message(bot, message.chat.id, "Вы не написали feedback!")
        return

    feedback_text = split_text[1:][0]

    with open(config.path_to_work_dir + config.path_to_reports, 'a', encoding='utf-8-sig') as file:
        username = message.from_user.username
        username = username if username is not None else message.from_user.last_name + ' ' + message.from_user.first_name

        file.write(username + ': ' + feedback_text[:300] + '\n')

    send_message(bot, message.chat.id, 'Спасибо за feedback!')


@bot.message_handler(content_types=['text'])
def moderate(message: Message):
    if check_the_message_is_not_from_the_group(message):
        return

    chat_id = str(message.chat.id)
    # get user
    user: User = message.from_user

    # load users data
    data = load_data(config.path_to_work_dir + config.path_to_json)

    if chat_id not in data:
        data = add_chat(chat_id, data)

    # find user in data
    user_index = -1
    if user.id in data[chat_id]['user_id']:
        for i, user_id in enumerate(data[chat_id]['user_id']):
            if user.id == user_id:
                user_index = i
                break
    else:
        # create user in data
        data = create_user(user.id, chat_id, data)
        user_index = len(data[chat_id]['user_id']) - 1

    # check the user for toxicity and change rating
    if check_is_toxic(message.text):
        data[chat_id]['rating'][user_index] -= config.fine_for_toxic
        data[chat_id]['toxic'][user_index] += 1
    else:
        data[chat_id]['rating'][user_index] += config.reward_for_positive
        data[chat_id]['positive'][user_index] += 1

    # check that the rating has not exceeded the threshold
    if data[chat_id]['rating'][user_index] < config.user_toxicity_threshold and not data[chat_id]['is_toxic'][
        user_index]:
        waring_text = 'очень токсичен. \nЧтобы узнать список токсичных людей, пропишите в чате /get_toxics'
        if data[chat_id]['ban_mode']:
            # ban toxic user
            waring_text = 'был забанен за токсичность'

            data = delete_user(user_index, chat_id, data)

            try:
                bot.kick_chat_member(message.chat.id, user.id)
                send_message(bot, message.chat.id, f'Пользователь @{user.username} {waring_text}')
            except Exception:
                send_message(bot, message.chat.id,
                             f'Я не могу банить пользователей. Дайте мне админ права или пропишите /set_ban_mode 0')
        else:
            # set that the user is toxic
            data[chat_id]['is_toxic'][user_index] = True

        for admin_id in bot.get_chat_administrators(message.chat.id):
            send_message(bot, admin_id.user.id, f'Warning: Пользователь @{user.username} {waring_text}')

    elif data[chat_id]['rating'][user_index] > config.user_toxicity_threshold and data[chat_id]['is_toxic'][user_index]:
        # set that the user is not toxic
        data[chat_id]['is_toxic'][user_index] = False

    # save user data
    save_data(data, config.path_to_work_dir + config.path_to_json)


bot.infinity_polling(timeout=15)
