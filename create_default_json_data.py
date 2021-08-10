import json

data = {
    'chat_id_example':{'admin_id':[],'user_id':[],'rating':[],'toxic':[],'positive':[],'is_toxic':[],'ban_mode': False}
}

with open('users.json', 'w') as file:
    file.write(json.dumps(data))