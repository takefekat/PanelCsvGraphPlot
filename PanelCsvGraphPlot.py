import csv
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os


class PanelCsvGraphPlot:

    # コンストラクタ
    def __init__(self):
        self.indir = "./IN/"
        self.outdir = "./OUT/"
        self.csv_files = glob.glob("./IN/*")
        self.data_name = os.path.commonprefix(self.csv_files)[5:]
        self.compress_file = "./OUT/" + self.data_name + "_compress.csv"
        return

    # keysで指定された列のみを取得
    def compress(self, keys):

        if os.path.exists(self.compress_file):
            print("Already exist :" + self.compress_file)
            return self.compress_file

        dfs = []
        for csv_file in self.csv_files:
            df = pd.read_csv(csv_file, usecols=keys, skiprows=4)
            dfs.append(df)
        df_concat = pd.concat(dfs)
        df_concat.to_csv(self.compress_file, index=False)

        print("Save compressed csv:" + self.compress_file)
        return self.compress_file

    def plot(self, csv_file):
        df = pd.read_csv(csv_file)
        l = df.T.values.tolist()

        # figure()でグラフを表示する領域をつくり，figというオブジェクトにする．
        fig = plt.figure(figsize=(50, (len(l)-1)*4))

        # add_subplot()でグラフを描画する領域を追加する．引数は行，列，場所
        ax = []
        for i in range(len(l)-1):
            ax.append(fig.add_subplot(len(l), 1, i+1))

        for i in range(len(l)):
            if i == 0:
                continue
            ax[i-1].plot(l[0], l[i], label=df.columns.values[i])
            ax[i-1].legend(bbox_to_anchor=(0, 1),
                           loc='center left', borderaxespad=0, fontsize=11)

        # plt.show()
        out_fig = self.outdir + self.data_name + ".png"
        fig.savefig(out_fig, bbox_inches='tight')
        return


def main():

    panel_data = PanelCsvGraphPlot()

    # keyによる圧縮
    keys = ['#data1', '#data2', '#data3', '#data4', '#data5',
            '#data6', '#data7', '#data8', '#data9', '#data10',
            '#data11', '#data12', '#data13', '#data14', '#data15']
    compress_file = panel_data.compress(keys)

    # グラフ描画
    panel_data.plot(compress_file)


def hoge():
    import numpy as np
    import matplotlib.cm as cm

    X = 10*np.random.rand(5, 3)

    fig = plt.figure(figsize=(15, 5), facecolor='w')
    ax = fig.add_subplot(111)
    ax.imshow(X, cmap=cm.jet)

    plt.savefig("image.png", bbox_inches='tight', dpi=100)


def example_data():
    row = [0, 0, 0, 0, 1]
    for i in range(1000000):
        print(i, ",", row, ",", row, ",", row)
        row[0] += 1

        if(i % 100 == 0):
            row[1] += 1

        row[2] += 1
        row[2] %= 50

        import random
        row[3] = random.randint(1, 100)
    return


if __name__ == "__main__":
    main()
    # example_data()
    # hoge()
