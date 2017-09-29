import pandas as pd


def Gordon_RSRF():
    """
    Read raw files
    """
    fns = ['RSRF/MIPS24_resp.dat', 'RSRF/MIPS70_resp.dat',
           'RSRF/MIPS160_resp.dat', 'RSRF/PACS70_resp.dat']
    titles = ['MIPS_24', 'MIPS_70', 'MIPS_160', 'PACS_70']
    assert len(fns) == len(titles)
    for i in range(len(fns)):
        df = pd.read_csv(fns[i])
        df.columns = ['Raw']
        for j in range(len(df)):
            temp = df.iloc[j]['Raw'].strip(' ').split(' ')
            df.set_value(j, 'Wavelength', float(temp[0]))
            df.set_value(j, titles[i], float(temp[-1]))
        del df['Raw']
        df.to_csv('RSRF/' + titles[i] + '.csv', index=False)


if __name__ == "__main__":
    Gordon_RSRF()
