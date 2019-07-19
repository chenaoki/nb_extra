import numpy as np
import scipy.interpolate as interpolate
from scipy.ndimage.filters import gaussian_filter1d
from scipy.signal import hilbert
from f_peakdetect import peakdetect

def f_peaks_line(ts,heading = 50):
    try:
        peaks, bottoms = peakdetect(ts, lookahead=heading)
        peaks = np.array(peaks)
        bottoms = np.array(bottoms)
        start = np.array([[0, ts[0]]])
        end = np.array([[len(ts), ts[-1]]])
        peaks_ = np.concatenate((start,peaks, end))
        bottoms_ = np.concatenate((start,bottoms, end))

        f = interpolate.PchipInterpolator(peaks_[:,0], peaks_[:,1])
        _peaks_ = f(np.arange(len(ts)))
        f = interpolate.PchipInterpolator(bottoms_[:,0], bottoms_[:,1])
        _bottoms_ = f(np.arange(len(ts)))

        return (_peaks_, _bottoms_)

    except:
        return (np.ones_like(ts)*np.mean(ts),np.ones_like(ts)*np.mean(ts))
def f_apd(ts,apd_thre = 90,heading = 50):
    #apd_thre = 90
    peaks,bottoms = f_peaks_line(ts,heading)
    #peaks = _peaks_
    #bottoms = _bottoms_
    peaks = np.array(peaks)
    bottoms = np.array(bottoms)
    apd_thre_line = peaks*(1-apd_thre/100.0) + bottoms*apd_thre/100.0
    apd_data = np.zeros_like(ts)
    upto = 1
    downto = 0
    apd_ct = 0
    apd_start = 0
    apd_end = 0
    delta = 0.05
    apd = 0
    for t,v in enumerate(ts):
        apd_data[t] = apd
        if t == 0:
            continue
        if upto ==1 and v > apd_thre_line[t]:
            apd_start = t
            upto = 0
            #apd_data[int(apd_end):int(apd_start)] = (apd_end - apd_start)*apd_ct
        if upto == 0 and v == peaks[t]:
            downto = 1
        if downto == 1 and v < apd_thre_line[t]:
            apd_end = t
            downto = 0
            apd = (apd_end - apd_start)*apd_ct
            apd_data[int(apd_start):int(apd_end)] = (apd_end - apd_start)*apd_ct
            apd_data[t] = apd
        if downto == 0 and v == bottoms[t]:
            upto = 1
            apd_ct = 1
    return apd_data
def f_apd_up2(ts,apd_thre = 90,heading = 50):
    #apd_thre = 90
    peaks,bottoms = f_peaks_line(ts,heading)
    peaks = np.array(peaks)
    bottoms = np.array(bottoms)
    up_thre = 0.6
    apd_thre_line = peaks*(1-apd_thre/100.0) + bottoms*apd_thre/100.0
    up_line = peaks*(1-up_thre/100.0) + apd_thre_line*up_thre/100.0
    apd_data = np.zeros_like(ts)
    upto = 0
    apd_start = 0
    apd_end = 0
    apd = 0
    for t,v in enumerate(ts):
        apd_data[t] = apd
        if t == 0:
            continue
        if v > apd_thre_line[t] and ts[t-1] < apd_thre_line[t-1]:
            upto = 1
            #apd_data[int(apd_end):int(apd_start)] = (apd_end - apd_start)*apd_ct
        if upto == 1 and v > up_line[t]:
            upto = 0
            apd_start = t
        elif upto == 0 and v < apd_thre_line[t]:
            apd_end = t
            upto = 1
            if apd_start == 0:
                continue
            apd = (apd_end - apd_start)
            apd_data[int(apd_start):int(apd_end)] = apd
            apd_data[t] = apd
    return apd_data
def f_apd_up(ts,apd_thre = 90,heading = 50):
    #apd_thre = 90
    peaks,bottoms = f_peaks_line(ts,heading)
    peaks = np.array(peaks)
    bottoms = np.array(bottoms)
    up_thre = 0.6
    apd_thre_line = peaks*(1-apd_thre/100.0) + bottoms*apd_thre/100.0
    up_line = peaks*up_thre + apd_thre_line*(1-up_thre)
    apd_data = np.zeros_like(ts)
    upto = 0
    apd_start = 0
    apd_end = 0
    apd = 0
    for t,v in enumerate(ts):
        apd_data[t] = apd
        if t == 0:
            continue
        if v > apd_thre_line[t] and ts[t-1] < apd_thre_line[t-1]:
            upto = 1
            #apd_data[int(apd_end):int(apd_start)] = (apd_end - apd_start)*apd_ct
        if upto == 1 and v > up_line[t]:
            upto = 0
            apd_start = t
        elif upto == 0 and v < apd_thre_line[t]:
            apd_end = t
            upto = 1
            if apd_start == 0:
                continue
            apd = (apd_end - apd_start)
            apd_data[int(apd_start):int(apd_end)] = apd
            apd_data[t] = apd
    return apd_data

def f_apd_map(ts,start = 0,end = None):
    start = 0
    if not end == end:
        end = len(ts)
    sample = ts[start:end]
    d_arr = np.zeros_like(ts)
    d_arr = d_arr[:12]
    apd = 0
    act = 0
    num = 0
    for kx,ap in enumerate(sample):
        if act != 0:
            if ap != apd:
                d_arr[num] = ap*act
                act *= -1
                num += 1
                if num%3 == 2:
                    d_arr[num] = d_arr[num-1]+d_arr[num-2]
                    num += 1
        if act == 0:
            if ap != 0:
                act = 1
        apd = ap
        if num == 12:
            break
    return d_arr

def f_iso(ts,heading = 50):
    peaks,bottoms = f_peaks_line(ts,heading)
    #peaks = _peaks_
    #bottoms = _bottoms_
    peaks = np.array(peaks)
    bottoms = np.array(bottoms)
    iso_data = np.zeros_like(ts)
    upto = 0
    iso_ct = 0
    st = 0
    up_thre = 0.7
    down_thre= 0.3
    up_thre_line = peaks*up_thre + bottoms*(1-up_thre)
    down_thre_line = peaks*down_thre + bottoms*(1-down_thre)

    for t,v in enumerate(ts):
        if t == 0:
            continue
        iso_ct += 1*st
        if (v > up_thre_line[t] and upto == 0):
            st = 1
            upto = 1
            iso_ct = 0
        if (v < down_thre_line[t] and upto == 1):
            upto = 0
        iso_data[t] = iso_ct
    return iso_data

def f_iso_up(ts,heading = 50,up_thre = 0.7,down_thre = 0.3):
    peaks,bottoms = f_peaks_line(ts,heading)
    #peaks = _peaks_
    #bottoms = _bottoms_
    peaks = np.array(peaks)
    bottoms = np.array(bottoms)
    iso_data = np.zeros_like(ts)
    upto = 0
    iso_ct = 0
    st = 0
    up_thre = 0.7
    down_thre= 0.3
    up_thre_line = peaks*up_thre + bottoms*(1-up_thre)
    down_thre_line = peaks*down_thre + bottoms*(1-down_thre)

    for t,v in enumerate(ts):
        if t == 0:
            continue
        iso_ct += 1*st
        if (v > up_thre_line[t] and upto == 0):
            st = 1
            upto = 1
            iso_ct = 1
            iso_data[t-1] = 0
        if (v < down_thre_line[t] and upto == 1):
            upto = 0
        iso_data[t] = iso_ct
    return iso_data

def stdsig(ts):
    if np.max(ts) == 0:
        return np.zeros_like(ts)
    maxsig = np.max(ts)
    minsig = np.min(ts)
    sig = 2*(ts-minsig)/(maxsig-minsig)-1
    return sig
