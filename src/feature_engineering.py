# === Importing Libraries ==============================

import numpy as np 

import pandas as pd 


# === Feature Engineering ============================== 

def feature_engineer(data):

    data = data.copy()

    

    data['Missing RAM Speed'] = data['RAM Speed (MHz)'].isna().astype(int)
    data['Missing Refresh Rate'] = data['Refresh Rate (Hz)'].isna().astype(int)



    data['CPU Brand'] = pd.Series(np.nan, index=data.index, dtype=object)

    data.loc[data['Processor'].str.lower().str.contains(r'\bintel\b|\binte core\b', na=False, regex=True), 'CPU Brand'] = 'intel'
    data.loc[data['Processor'].str.lower().str.contains(r'\bamd\b|\bryzen\b|\bapu\b|\bathlon\b|\bradeon\b', na=False, regex=True), 'CPU Brand'] = 'amd'
    data.loc[data['Processor'].str.lower().str.contains(r'\bapple\b', na=False, regex=True), 'CPU Brand'] = 'apple'
    data.loc[data['Processor'].str.lower().str.contains(r'\bqualcomm\b|\bsnapdragon\b|\bmicrosoft\b', na=False, regex=True), 'CPU Brand'] = 'qualcomm'
    data.loc[data['Processor'].str.lower().str.contains(r'\bmediatek\b', na=False, regex=True), 'CPU Brand'] = 'mediatek'



    data['CPU Segment'] = pd.Series(np.nan, index=data.index, dtype=object)

    norm_cpu = data['Processor'].str.lower().str.replace('-', ' ', regex=False).str.replace(r'\s+', ' ', regex=True).str.strip()

    low_tier_cpu = [
        r'core i3\b', r'\bcore 3\b', r'\bcore m3\b', r'\bcore m5\b',
        r'celeron', r'pentium', r'\batom\b',
        r'\ba4\b', r'\ba6\b', r'\ba8\b', r'\ba9\b', r'\ba10\b', r'\ba12\b',
        r'\be1\b', r'\be2\b', r'\bfx\b',
        r'ryzen 3\b', r'athlon(?! gold)(?! silver)', r'athlon gold', r'athlon silver',
        r'apu\b(?!.*ryzen)',
        r'snapdragon 7c', r'snapdragon 665', r'snapdragon 835',
        r'kompanio', r'\bmt8', r'microsoft sq1',
        r'p60t', r'athon',
        r'dual core 3020e', r'\bi3\b'
    ]

    mid_tier_cpu = [
        r'core i5\b', r'\bcore 5\b', r'core ultra 5', r'ryzen 5\b', r'\bi5\b',
        r'snapdragon x\b', r'snapdragon x plus',
        r'^apple m1$', r'^apple m2$', r'^apple m3$', r'^apple m3 chip$', r'^apple m4$'
    ]

    high_tier_cpu = [
        r'core i7\b', r'core i9\b', r'\bcore 7\b', r'core ultra 7', r'core ultra 9',
        r'ryzen 7\b', r'ryzen 9\b', r'ryzen ai 7', r'ryzen ai 9',
        r'xeon', r'xenon',
        r'\bi7\b', r'\bi9\b',
        r'snapdragon x elite', r'snapdragon.*835',
        r'\bm1 pro\b', r'\bm1 max\b', r'\bm2 pro\b', r'\bm2 max\b',
        r'\bm3 pro\b', r'\bm3 max\b', r'\bm4 pro\b', r'\bm4 max\b'
    ]

    data.loc[norm_cpu.str.contains('|'.join(low_tier_cpu), na=False, regex=True), 'CPU Segment'] = 'low'
    data.loc[norm_cpu.str.contains('|'.join(mid_tier_cpu), na=False, regex=True), 'CPU Segment'] = 'mid'
    data.loc[norm_cpu.str.contains('|'.join(high_tier_cpu), na=False, regex=True), 'CPU Segment'] = 'high'



    data['CPU Series'] = pd.Series(np.nan, index=data.index, dtype=object)
    
    low_series = [
        r'\d+y\b', r'\d+u\b', r'\d+ug\d\b', r'\d+g\d\b', r'\d+g\b', r'\bn\d{3,4}\b',
        r'z\d{3,4}[a-z]?\b', r'\d+e\b', r'\d+c\b', r'\d+v\b', r'\d+ng\d\b',
        r'\d+du\b', r't\d{4}\b',
        r'mediatek',
        r'qualcomm|snapdragon|microsoft'
    ]

    mid_series = [
        r'\d+p\b',
        r'snapdragon.*plus', 
        r'^apple'
    ]

    high_series = [
        r'\d+h[skxfqi]?\b',
        r'\bhx\b',     
        r'\d+mq\b',
        r'snapdragon.*elite',
        r'apple.*pro|apple.*max'
    ]

    data.loc[data['Processor'].str.lower().str.contains('|'.join(low_series), na=False, regex=True), 'CPU Series'] = 'low'
    data.loc[data['Processor'].str.lower().str.contains('|'.join(mid_series), na=False, regex=True), 'CPU Series'] = 'mid'
    data.loc[data['Processor'].str.lower().str.contains('|'.join(high_series), na=False, regex=True), 'CPU Series'] = 'high'


    data.drop(columns=['Processor'], inplace=True)

    

    data['GPU Brand'] = pd.Series(np.nan, index=data.index, dtype=object)

    data.loc[data['Graphic Processor'].str.lower().str.contains(r'\bnvidia\b|\bgeforce\b|\bnividia\b', na=False, regex=True), 'GPU Brand'] = 'nvidia'
    data.loc[data['Graphic Processor'].str.lower().str.contains(r'\bamd\b|\bradeon\b|\bati\b', na=False, regex=True), 'GPU Brand'] = 'amd'
    data.loc[data['Graphic Processor'].str.lower().str.contains(r'\bintel\b', na=False, regex=True), 'GPU Brand'] = 'intel'
    data.loc[data['Graphic Processor'].str.lower().str.contains(r'\bapple\b', na=False, regex=True), 'GPU Brand'] = 'apple'
    data.loc[data['Graphic Processor'].str.lower().str.contains(r'\bqualcomm\b|\badreno\b', na=False, regex=True), 'GPU Brand'] = 'qualcomm'
    data.loc[data['Graphic Processor'].str.lower().str.contains(r'\bmediatek\b|\barm\b', na=False, regex=True), 'GPU Brand'] = 'mediatek'

    

    data['GPU Series'] = pd.Series(np.nan, index=data.index, dtype=object)

    norm_gpu = data['Graphic Processor'].str.lower().str.replace(r'[\s\-]+', '', regex=True).str.strip()

    low_tier_gpu = [
        'intelhd', 'inteluhd', 'intelgfx', 'intelgraphics', 'intelintegrated', 'inteluma', 'iris', '^intel$',
        'radeongraphics', 'amdgraphics', 'amdintegrated', 'radeonhd', 'amdradeonhd', 'radeonr2', 'radeonr3', 'radeonr4', 'radeonr5', 'radeonr7', 'athlon', 'vega2', 'vega3', 'vega5', 'vega6', 'vega7', 'vega8', 'vega10', 'vega$', 'radeon610', 'radeon430', 'radeon520', 'radeon530', 'radeon535', 'radeon540', 'radeon3250u', 'radeon$', 'radeonuhd', '940mx', 'r17m', 'r16m', 'atiexopro', 'atimobility',
        'geforce8', 'geforce92', 'geforce93', 'geforce94', 'geforcegt7', 'geforcegt8', 'geforcegt9',
        'mx110', 'mx130', 'mx150', 'mx230', 'mx250', 'mx330',
        'gtx850m', 'gtx860m', 'gtx960', 'gtx980m', 'gtx1050', 'gtx1650',
        'rtx2050', 'rtx1650',
        'quadrot50$', 'quadromx130',
        'mali', 'adreno', 'mediatek', 'n16s', 'n16v',
        'appleintegrated', 'qualcommintegrated'
    ]

    mid_tier_gpu = [
        'mx350', 'mx450', 'mx550', 'mx570',
        'gtx1060', 'gtx1660','geforce1060', 'geforce1650',
        'gtx2060', 'gtx3050', 'gtx3060',
        'rtx2060', 'rtx3050', 'rtx3060', 'rtx4050', 'rtx4060', 'rtx3000',
        'rtxa1000', 'rtxa2000', 'rtxa500', 'nvidiaa2000',
        'quadrot5', 'quadrot6', 'quadrot1', 'quadrot2', 't500', 't550', 't600', 't1200', 'quadrop1000', 'quadrop620', 'geforcep620', 'quadrortx3000',
        'rx550', 'rx560', 'rx580', 'rx640', 'rx5500m', 'rx5600m','rx6500m', 'rx6550m', 'rx6600m', 'rx6650m',
        'radeon660m', 'radeon680m', 'radeon740m', 'radeon760m', 'radeon780m', 'radeon860m', 'radeon880m', 'radeon890m', 'radeonpro', 'vegamgl',
        'arc130v', 'arc140v', 'arc140t', 'arca370m', 'arca530m', '^intelarc$', 'intelintegratedarc', 'intelintegratedirisxe', 'irisxe',
        'applem1', '^applem2$', 'applem2integrated', '^applem3$', '^applem4$'
    ]

    high_tier_gpu = [
        'gtx1070', 'gtx1080', 'rtx2070', 'rtx2080', 'rtx3070', 'rtx3080',
        'rtx4070', 'rtx4080', 'rtx4090', 'rtx5070', 'rtx5080', 'rtx5090',
        'rtx1000ada', 'rtx2000ada', 'rtx3000ada', 'rtx3500ada', 'rtx4000ada', 'rtx500ada', 'rtx5000ada',
        'quadrortx5000',
        'rx6700m', 'rx6700s', 'rx6800m', 'rx6800s', 'rx7600s',
        'applem2pro', 'applem2max', r'apple\d+coregpu',
    ]

    data.loc[norm_gpu.str.contains('|'.join(low_tier_gpu), na=False, regex=True), 'GPU Series'] = 'low'
    data.loc[norm_gpu.str.contains('|'.join(mid_tier_gpu), na=False, regex=True), 'GPU Series'] = 'mid'
    data.loc[norm_gpu.str.contains('|'.join(high_tier_gpu), na=False, regex=True), 'GPU Series'] = 'high'
    

    data.drop(columns=['Graphic Processor'], inplace=True)


    
    return data