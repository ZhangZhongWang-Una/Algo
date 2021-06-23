import pandas as pd
import json
import os

USER_ACTION = '../../data/v1/origin/user_action.csv'
FEED_INFO = '../../data/v1/origin/feed_info.csv'
TEST_A = '../../data/v1/origin/test_a.csv'
ROOT_PATH = '../../data/v4/'
FEA_COLUMN_LIST = ["read_comment", "like", "click_avatar",  "forward"]
# user_action = pd.read_csv(USER_ACTION)
# feed_info = pd.read_csv(FEED_INFO)
# manual_keyword_list = feed_info['manual_keyword_list']
# machine_tag_list = feed_info['machine_tag_list']

statis = json.loads(json.load(open(os.path.join(ROOT_PATH, 'statis.json'))))
tmp = int(statis['description_map_len'])
print(1)

