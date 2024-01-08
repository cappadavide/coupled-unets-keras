import pandas as pd
import numpy as np

def saveAccsCSV(modelParams):
    accs = np.load(f"../cunet{modelParams['nUNet']}_{modelParams['m']}{modelParams['n']}_{len(modelParams['interSupervisions'])}S_accs.npz")['arr_0']
    df = pd.DataFrame(accs)
    df.columns = ['acc']
    df['acc'] = df['acc'].apply(lambda x: '{:.5f},'.format(x))
    df['acc'] = df['acc'].str.replace('.', ',', regex=True)
    df['acc'] = df['acc'].str.rstrip(',')

    df.to_csv(f"../cunet{modelParams['nUNet']}_{modelParams['m']}{modelParams['n']}_{len(modelParams['interSupervisions'])}S_accs.csv", index=False)

