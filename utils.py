from telebot.types import Message

__all__ = ['check_is_admin', 'check_the_message_is_not_from_the_group', 'send_message']

# check the message is not from the group
def check_the_message_is_not_from_the_group(message: Message) -> bool:
    if message.chat.type != 'group' and message.chat.type != 'supergroup':
        return True
    return False


# check that the user is the admin of the group
def check_is_admin(client, user_id: int, chat_id: int) -> bool:
    for admin in client.get_chat_administrators(chat_id):
        if admin.user.id == user_id:
            return True
    return False


# send message
def send_message(client, recipient_id: int, msg_text: str):
    try:
        client.send_message(recipient_id, msg_text)
    except Exception:
        pass