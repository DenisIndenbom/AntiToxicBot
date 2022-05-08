import config

import json
import copy

# load data from json
def load_data(path: str) -> dict:
    data = None
    try:
        with open(path, 'r') as file:
            data = json.loads(file.read())
    except Exception as e:
        if e == json.JSONDecodeError:
            print('Data is crash')
            exit(1)

    return data


# save data to json
def save_data(data: dict, path: str) -> None:
    with open(path, 'w') as file:
        file.write(json.dumps(data))


# add chat to data
def add_chat(data: dict, chat_id: str) -> dict:
    data[chat_id] = copy.deepcopy(data['chat_id_example'])
    data[chat_id]['users'].pop('user_id_example')

    return data


# delete chat
def delete_chat(data: dict, chat_id: str) -> dict:
    data.pop(chat_id)

    return data


# return true if chat exist else false
def chat_exists(data: dict, chat_id: str) -> bool:
    return chat_id in data


# get all members of chat
def get_chat_users(data: dict, chat_id: str):
    return data[chat_id]['users']


# get chat ban mode
def get_chat_ban_mode(data: dict, chat_id: str) -> bool:
    return data[chat_id]['ban_mode']


# set chat ban mode
def set_chat_ban_mode(data: dict, chat_id: str, mode: bool) -> dict:
    data[chat_id]['ban_mode'] = mode

    return data


# create user in the data
def create_user(data: dict, chat_id: str, user_id: str) -> dict:
    # create user
    data[chat_id]['users'][user_id] = copy.deepcopy(data['chat_id_example']['users']['user_id_example'])
    # return data
    return data


# delete user from the data
def delete_user(data: dict, chat_id: str, user_id: str) -> dict:
    # delete user
    data[chat_id]['users'].pop(user_id)
    # return data
    return data


# return true if user exist else false
def user_exists(data: dict, chat_id: str, user_id: str) -> bool:
    return user_id in data[chat_id]['users']


# get user rating in chat
def get_user_rating(data: dict, chat_id: str, user_id: str) -> float:
    return data[chat_id]['users'][user_id]['rating']


# get user num of toxic messages in chat
def get_user_toxic(data: dict, chat_id: str, user_id: str) -> float:
    return data[chat_id]['users'][user_id]['toxic']


# get user num of positive messages in chat
def get_user_positive(data: dict, chat_id: str, user_id: str) -> float:
    return data[chat_id]['users'][user_id]['positive']


# get user toxic state
def get_user_toxic_state(data: dict, chat_id: str, user_id: str) -> bool:
    return data[chat_id]['users'][user_id]['is_toxic']


# change user rating
def change_user_rating(data: dict, chat_id: str, user_id: str, is_toxic: bool) -> dict:
    if is_toxic:
        data[chat_id]['users'][user_id]['rating'] -= config.fine_for_toxic
        data[chat_id]['users'][user_id]['toxic'] += 1
    else:
        data[chat_id]['users'][user_id]['rating'] += config.reward_for_positive
        data[chat_id]['users'][user_id]['positive'] += 1

    return data


# set user toxic or not toxic
def set_user_toxic_status(data: dict, chat_id: str, user_id: str, is_toxic: bool) -> dict:
    data[chat_id]['users'][user_id]['is_toxic'] = is_toxic

    return data