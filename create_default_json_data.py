import json

if input('Ты правда хочешь пересоздать дату? (y/n)\n').lower() == 'y':
    data = {
        'chat_id_example':{'users':{'user_id_example':{'rating': 0, 'toxic': 0, 'positive': 0, 'is_toxic': False}}, 'ban_mode': False}
    }

    with open('users.json', 'w') as file:
        file.write(json.dumps(data))
    print('Операция завершена')
else:
    print('Операция отменена')