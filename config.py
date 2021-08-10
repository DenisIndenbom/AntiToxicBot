token = '1925678461:AAGY6D-Rq3kmaSifnx_oZ5wZdQBGgKeqTc0'

toxic_threshold = -5

with open('bad_words.txt', 'r', encoding='utf-8-sig') as file:
    bad_words = file.read().split()

fine_for_toxic = 0.5

reward_for_positive = 0.25