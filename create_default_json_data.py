import json

if input('Ты правда хочешь пересоздать дату? (y/n)\n').lower() == 'y':
    data = {
        'chat_id_example':{'user_id':[],'rating':[],'toxic':[],'positive':[],'is_toxic':[],'ban_mode': False}
    }

    with open('users.json', 'w') as file:
        file.write(json.dumps(data))
    print('Операция завершена')
else:
    print('Операция отменена')