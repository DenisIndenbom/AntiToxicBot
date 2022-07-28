import config

import json
import copy
import time
from datetime import datetime

from threading import Lock

__all__ = ['UserDataWorker']

def lock(function):
    def wrapper(self, *args, **kwargs):
        with self.lock:
            result = function(self, *args, **kwargs)

        return result

    return wrapper


class UserDataWorker:
    def __init__(self, path_to_work_dir, path_to_file):
        self.data = self.load_data(path_to_work_dir + path_to_file)

        self.work_dir = path_to_work_dir

        self.lock = Lock()

    @staticmethod
    def load_data(path: str) -> dict:
        """
        Load data from json
        :param path:
        :return dict with data:
        """
        data = None
        try:
            with open(path, 'r') as file:
                data = json.loads(file.read())
        except Exception as e:
            if e == json.JSONDecodeError:
                print('Data is crash')
                exit(1)

        return data

    @lock
    def save_data(self, path: str) -> None:
        """
        Save data to json
        :param path: string
        :return: none
        """
        with open(self.work_dir + path, 'w') as file:
            file.write(json.dumps(self.data))

    @lock
    def add_chat(self, chat_id: str) -> dict:
        """
        Add chat to data
        :param chat_id: sting
        :return data: dict
        """
        self.data[chat_id] = copy.deepcopy(self.data['chat_id_example'])
        self.data[chat_id]['users'].pop('user_id_example')

        return self.data

    @lock
    def delete_chat(self, chat_id: str) -> dict:
        """
        Delete chat
        :param chat_id: string
        :return data: dict
        """
        self.data.pop(chat_id)

        return self.data

    @lock
    def chat_exists(self, chat_id: str) -> bool:
        """
        Return true if chat exist else false
        :param chat_id: string
        :return result: bool
        """
        return chat_id in self.data

    @lock
    def get_chat_users(self, chat_id: str):
        """
        Get all members of chat
        :param chat_id: string
        :return members:
        """
        return self.data[chat_id]['users']

    @lock
    def get_chat_ban_mode(self, chat_id: str) -> bool:
        """
        Get chat ban mode
        :param chat_id: string
        :return: bool
        """
        return self.data[chat_id]['ban_mode']

    @lock
    def set_chat_ban_mode(self, chat_id: str, mode: bool) -> dict:
        """
        Set chat ban mode
        :param chat_id: string
        :param mode: bool
        :return data: dict
        """
        self.data[chat_id]['ban_mode'] = mode

        return self.data

    @lock
    def create_user(self, chat_id: str, user_id: str) -> dict:
        """
        Create user in the data
        :param chat_id: string
        :param user_id: string
        :return data: dict
        """
        # create user
        self.data[chat_id]['users'][user_id] = copy.deepcopy(self.data['chat_id_example']['users']['user_id_example'])
        # return data
        return self.data

    @lock
    def delete_user(self, chat_id: str, user_id: str) -> dict:
        """
        Delete user from the data
        :param chat_id: string
        :param user_id: string
        :return data: dict
        """
        # delete user
        self.data[chat_id]['users'].pop(user_id)
        # return data
        return self.data

    @lock
    def user_exists(self, chat_id: str, user_id: str) -> bool:
        """
        Return true if user exist else false
        :param chat_id:
        :param user_id:
        :return result: bool
        """
        return user_id in self.data[chat_id]['users']

    @lock
    def get_user_rating(self, chat_id: str, user_id: str) -> float:
        """
        Get user rating in chat
        :param chat_id: string
        :param user_id: string
        :return rating: float
        """
        return self.data[chat_id]['users'][user_id]['rating']

    @lock
    def get_user_toxic(self, chat_id: str, user_id: str) -> float:
        """
        Get user num of toxic messages in chat
        :param chat_id: string
        :param user_id: string
        :return num of toxic messages: float
        """
        return self.data[chat_id]['users'][user_id]['toxic']

    @lock
    def get_user_positive(self, chat_id: str, user_id: str) -> float:
        """
        Get user num of positive messages in chat
        :param chat_id: string
        :param user_id: string
        :return num of positive messages: float
        """
        return self.data[chat_id]['users'][user_id]['positive']

    @lock
    def get_user_toxic_state(self, chat_id: str, user_id: str) -> bool:
        """
        Get user toxic state
        :param chat_id: string
        :param user_id: string
        :return state: bool
        """
        return self.data[chat_id]['users'][user_id]['is_toxic']

    @lock
    def change_user_rating(self, chat_id: str, user_id: str, is_toxic: bool) -> dict:
        """
        Change user rating
        :param chat_id: string
        :param user_id: string
        :param is_toxic: bool
        :return data: dict
        """
        if is_toxic:
            self.data[chat_id]['users'][user_id]['rating'] -= config.fine_for_toxic
            self.data[chat_id]['users'][user_id]['toxic'] += 1
        else:
            self.data[chat_id]['users'][user_id]['rating'] += config.reward_for_positive
            self.data[chat_id]['users'][user_id]['positive'] += 1

        return self.data

    @lock
    def set_user_toxic_status(self, chat_id: str, user_id: str, is_toxic: bool) -> dict:
        """
        Set user toxic or not toxic
        :param chat_id: string
        :param user_id: string
        :param is_toxic: bool
        :return data: dict
        """
        self.data[chat_id]['users'][user_id]['is_toxic'] = is_toxic

        return self.data

    def make_backups(self, path: str, period: int = 30) -> None:
        """
        This function makes backups with a given gear ratio. Run in a separate process
        :param path: string
        :param period: int
        :return: None
        """
        while True:
            time.sleep(period * 60)
            self.save_data(path + "backup-" + datetime.now().strftime('%d_%m_%y-%I_%M') + '.json')
