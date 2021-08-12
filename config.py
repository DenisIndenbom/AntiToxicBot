toxic_threshold = -5

with open('bad_words.txt', 'r', encoding='utf-8-sig') as file:
    bad_words = file.read().split()

fine_for_toxic = 0.25

reward_for_positive = 0.25

NN_mode = True

GPU_mode = False