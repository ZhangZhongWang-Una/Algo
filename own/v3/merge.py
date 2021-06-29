import numpy as np
import pandas as pd
import time
nn_filename = r"./submit/submit_1624887693.csv"
tree_filename = r"./submit/lgb_1624887693.csv"
nn_weight = 0.55
tree_weight = 1 - nn_weight


if __name__ == '__main__':
    print('\033[32;1m[DATA]\033[0m start load data, please wait')
    T = time.time()

    nn_submit = np.loadtxt(nn_filename, comments='#', delimiter=",", skiprows=1, )
    tree_submit = np.loadtxt(tree_filename, comments='#', delimiter=",", skiprows=1, )

    print('\033[32;1m[DATA]\033[0m load data done, total cost {}'.format(
        time.strftime("%H:%M:%S", time.gmtime(float(time.time() - T)))))

    assert (nn_submit[:, 0] == tree_submit[:, 0]).all()
    assert (nn_submit[:, 1] == tree_submit[:, 1]).all()

    merge_submit_data = (nn_weight * nn_submit[:, 2:] + tree_weight * tree_submit[:, 2:])
    merge_submit = np.concatenate((nn_submit[:, :2], merge_submit_data), axis=1)
    merge_submit = pd.DataFrame(merge_submit)
    merge_filename = "./submit/merge_" + str(int(time.time())) + ".csv"
    merge_submit.to_csv(merge_filename, index=False,
                        header=['userid', "feedid", "read_comment", "like", "click_avatar", 'forward'])
    print('\033[32;1m[FILE]\033[0m save to {}'.format(merge_filename))