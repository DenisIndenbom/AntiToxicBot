import config

from user_data_worker import UserDataWorker

from utils import *

from toxicity_detector import NNClassifier

import telebot
from telebot.types import Message, User

import threading

import sentry_sdk

sentry_sdk.init(config.sentryToken, traces_sample_rate=0.35)

# init telegram bot
bot = telebot.TeleBot(token=config.telegram_token, threaded=True)

# init data worker
data_worker = UserDataWorker(config.path_to_work_dir, config.path_to_json)

# init toxicity classifier
classifier = NNClassifier(gpu=config.GPU_mode,
                          message_toxicity_threshold=config.message_toxicity_threshold,
                          model_path=config.path_to_model,
                          navec_path=config.path_to_navec)


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
def habr(message: Message) -> None:
    send_message(bot, message.chat.id, 'Первая статья на Хабре - https://habr.com/ru/post/582130/ \n'
                                       'Вторая статья на Хабре - https://habr.com/ru/post/652447/')


@bot.message_handler(commands=['reset_chat'])
def reset_chat(message: Message) -> None:
    if check_the_message_is_not_from_the_group(message):
        send_message(bot, message.chat.id, 'Эта команда работает только в группах')
        return

    chat_id = str(message.chat.id)

    if not data_worker.chat_exists(chat_id):
        return

    if not check_is_admin(bot, message.from_user.id, message.chat.id):
        send_message(bot, message.chat.id, f'@{message.from_user.username} вы не админ!')
        return

    data_worker.delete_chat(chat_id)
    send_message(bot, message.chat.id, 'Статистика чата сброшена!')
    data_worker.add_chat(chat_id)

    data_worker.save_data(config.path_to_json)


@bot.message_handler(commands=['set_ban_mode'])
def set_ban_mode(message: Message) -> None:
    if check_the_message_is_not_from_the_group(message):
        send_message(bot, message.chat.id, 'Эта команда работает только в группах')
        return

    chat_id = str(message.chat.id)

    if not data_worker.chat_exists(chat_id):
        data_worker.add_chat(chat_id)

    if not check_is_admin(bot, message.from_user.id, message.chat.id):
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

    data_worker.set_chat_ban_mode(chat_id, ban_mode)

    send_message(bot, message.chat.id, f'ban_mode {int(ban_mode)}')

    data_worker.save_data(config.path_to_json)


@bot.message_handler(commands=['get_statistics'])
def get_statistics(message: Message):
    if check_the_message_is_not_from_the_group(message):
        send_message(bot, message.chat.id, 'Эта команда работает только в группах')
        return

    chat_id = str(message.chat.id)

    if not data_worker.chat_exists(chat_id):
        send_message(bot, message.chat.id, 'Статистики пока нет')
        return

    users_stat = []

    for user_id in data_worker.get_chat_users(chat_id):
        try:
            user = bot.get_chat_member(message.chat.id, int(user_id)).user
            username = user.username if user.username is not None \
                else user.last_name + ' ' + user.first_name

            users_stat.append([username,
                               'rating ' + str(data_worker.get_user_rating(chat_id, user_id)),
                               'toxic ' + str(data_worker.get_user_toxic(chat_id, user_id)),
                               'positive ' + str(data_worker.get_user_positive(chat_id, user_id))])
        except Exception:
            pass

    statistics = ''
    for row in users_stat:
        buf = ''

        for item in row:
            buf += str(item) + ' | '

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

    if chat_id not in data_worker.data:
        send_message(bot, message.chat.id, 'Токсиков нет')
        return

    toxics = ''

    for user_id in data_worker.get_chat_users(chat_id):
        if data_worker.get_user_toxic_state(chat_id, user_id):
            try:
                # get user
                user = bot.get_chat_member(message.chat.id, int(user_id)).user
                # get username
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
        send_message(bot, message.chat.id, 'Список токсиков:')

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
    # get user id
    user_id = str(user.id)

    if not data_worker.chat_exists(chat_id):
        data_worker.add_chat(chat_id)

    if not data_worker.user_exists(chat_id, user_id):
        data_worker.create_user(chat_id, user_id)

    # check the user for toxicity and change rating
    data_worker.change_user_rating(chat_id, user_id, classifier.check_is_toxic(message.text))

    # check that the rating has not exceeded the threshold
    if data_worker.get_user_rating(chat_id, user_id) < config.user_toxicity_threshold and not \
            data_worker.get_user_toxic_state(chat_id, user_id):
        waring_text = 'очень токсичен. \nЧтобы узнать список токсичных людей, пропишите в чате /get_toxics'

        if data_worker.get_chat_ban_mode(chat_id):
            # ban toxic user
            ban_is_successful = True

            try:
                bot.kick_chat_member(message.chat.id, user.id)
                send_message(bot, message.chat.id, f'Пользователь @{user.username} {waring_text}')
            except Exception:
                send_message(bot, message.chat.id,
                             f'Я не могу банить пользователей. Дайте мне админ права или пропишите /set_ban_mode 0')
                ban_is_successful = False

            if ban_is_successful:
                data_worker.delete_user(chat_id, user_id)
            else:
                data_worker.set_user_toxic_status(chat_id, user_id, True)

            waring_text = 'был забанен за токсичность' if ban_is_successful else waring_text + '\n\nЯ не могу забанить пользователя. ' \
                                                                                               'Дайте админ права или пропишите команду /set_ban_mode 0 в чат'
        else:
            # set that the user is toxic
            data_worker.set_user_toxic_status(chat_id, user_id, True)

        for admin_id in bot.get_chat_administrators(message.chat.id):
            send_message(bot, admin_id.user.id, f'Warning: Пользователь @{user.username} {waring_text}')

    elif data_worker.get_user_rating(chat_id, user_id) > config.user_toxicity_threshold and \
            data_worker.get_user_toxic_state(chat_id, user_id):
        # set that the user is not toxic
        data_worker.set_user_toxic_status(chat_id, user_id, False)

    # save user data
    data_worker.save_data(config.path_to_json)


threading.Thread(target=data_worker.make_backups,
                 args=(config.path_to_backup_dir, config.periodic_backups)).start()

bot.infinity_polling(timeout=60)
