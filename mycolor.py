import numpy as np
from matplotlib.colors import LinearSegmentedColormap

def set_col_extra(eps = 1e-5):
    #[r,g,b]
    #ExTRaの位相マップ
    color = [
        (127,0,127), #-pi
        (63,0,127),
        (0,0,127),
        (0,63,127),
        (0,127,127),
        (0,127,0), 
        (0,159,0),
        (127,191,0),#この間が0
        (255,255,0),
        (255,223,0),
        (255,191,0),
        (255,127,0),
        (255,95,0),
        (255,0,0),
        (255,0,127),
        (191,0,127)#pi
    ]
    color_level = np.linspace(0,1,17)
    #eps = level#1e-5
    color_nm = []
    for i in range(len(color)):
        col0 = (color[i][0]/255.0,color[i][1]/255.0,color[i][2]/255.0)
        color_nm.append(col0)
    color_list = []
    for i in range(len(color_nm)):
        col1 = (color_level[i],color_nm[i])
        color_list.append(col1)
        if i != len(color_nm)-1:
            col1 = (color_level[i+1]-eps,color_nm[i])
            color_list.append(col1)
        else:
            col1 = (color_level[i+1],color_nm[i])
            color_list.append(col1)
    return LinearSegmentedColormap.from_list("extra",color_list)
