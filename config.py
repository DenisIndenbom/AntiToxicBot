import telegramBotAPIToken
import sentryToken

path_to_work_dir = ''

path_to_json = 'users.json'
path_to_reports = 'reports.txt'
path_to_help = 'help.txt'

path_to_backup_dir = 'backups/'

path_to_model = 'TextTonalityClassifier.nn'
path_to_navec = 'navec_hudlit_v1_12B_500K_300d_100q.tar'

periodic_backups = 30

user_toxicity_threshold = -5

message_toxicity_threshold = 0.7

fine_for_toxic = 0.25

reward_for_positive = 0.25

GPU_mode = True

telegram_token = telegramBotAPIToken.telegram_token
sentryToken = sentryToken.sentry_token
