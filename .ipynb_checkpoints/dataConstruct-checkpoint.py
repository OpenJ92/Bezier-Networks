import numpy as np
import pandas as pd

class dataConstruct():

    def __init__(self, file_path):
        self.file_path_ = file_path
        self.data_frame_ = self.pull()

        # self.labels_ = self.data_frame_['surface'].unique()
        self.u_sid_ = self.data_frame_['series_id'].unique()

    def run(self):
        pass

    def pull(self):
        return pd.read_csv(self.file_path_)

    def tensor_order_0(self):
        return np.stack([self.data_frame_[self.data_frame_['series_id'] == sid][self.data_frame_.columns[3:]] for sid in self.u_sid_], axis=2)

    #def tensor_order_1(self):
    #    to0 = self.tensor_order_0()
    #    to1 = np.zeros(shape = (to0.shape[2], to0.shape[1], to0.shape[0], to0.shape[0]))
    #    for k in range(to0.shape[2]):
    #        for i in range(to0.shape[0]):
    #            for j in range(to0.shape[0]):
    #                print(i, j, k)
    #                if i == j:
    #                    to1[k, :, i, j] = to0[i, :, k]
    #                else:
    #                    to1[k, :, i, j] = to0[i, :, k] - to0[j, :, k]

    #    return to1

    #def tensor_order_2(self):
    #    # We're going to have to wait or develop the Conv4D function in order to make use of this construct.
    #    # Additionally, this would take an extrodinary amount of time to fully construct. 
    #    to1 = self.tensor_order_1()
    #    to2 = np.zeros(shape = (to1.shape[0], to1.shape[1], to1.shape[2], to1.shape[2], to1.shape[2], to1.shape[2]))
    #    for k in range(to1.shape[0]):
    #        for i in range(to1.shape[2]):
    #            for j in range(to1.shape[2]):
    #                for m in range(to1.shape[2]):
    #                    for n in range(to1.shape[2]):
    #                        print(i, j, m, n, k)
    #                        if (i == j) and (m == n):
    #                            to2[k, :, i, j, m, n] = to1[k, :, i, j]
    #                        elif (i == j) or (m == n):
    #                            if i == j:
    #                                to2[k, :, i, j, m, n] = to1[k, :, m, n]
    #                            elif m == n:
    #                                to2[k, :, i, j, m, n] = to1[k, :, i, j]
    #                        else:
    #                            to2[k, :, i, j, m, n] = to1[k, :, i, j] - to1[k, :, m, n]

    #    return to2

    def tensor_order_T(self):
        to0 = self.tensor_order_0()
        toT = np.zeros(shape = (to0.shape[2], to0.shape[1], to0.shape[0], to0.shape[0]))

        toTs = self.seed_toT(to0, toT)

        for k in range(1, toTs.shape[2]):
            toTs = self.diag_toT(toTs, k)

        return toTs

    def seed_toT(self, to0, toT):
        for k in range(to0.shape[2]):
            for i in range(to0.shape[0]):
                toT[k, :, i, i] = to0[i, :, k]
        return toT

    def diag_toT(self, toTs, iIndex):
        for k in range(toTs.shape[0]):
            for i in range(toTs.shape[2] - iIndex):
                print()
                toTs[k, :, i, i + iIndex] = toTs[k, :, i, i + iIndex - 1] - toTs[k, :, i + 1, i + iIndex]
                toTs[k, :, i + iIndex, i] = -1 * toTs[k, :, i, i + iIndex]
        return toTs


if __name__ == '__main__':
    dC = dataConstruct('X_train/X_train.csv')
    # Look to reconstruct toT on meanshifted and scaled data. The values in nOrderDiff are astronomical.
    # to0 = dC.tensor_order_0()
    # to1 = dC.tensor_order_1()
    # toT = dC.tensor_order_T()
