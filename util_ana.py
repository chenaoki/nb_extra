import numpy as np
import matplotlib.pyplot as plt
import csv
import os
from mycolor import set_col_extra

def stdsig(ts):
    if np.max(ts) == 0:
        return np.zeros_like(ts)
    maxsig = np.max(ts)
    minsig = np.min(ts)
    sig = 2*(ts-minsig)/(maxsig-minsig)-1
    return sig

def labeling(pict,pos_h,pos_w,clabel):
    pict[pos_h,pos_w] = clabel
def labeling_im(img,img_ref,dx,dy,im_label):
    labeling(img,dx,dy,im_label)
    assert(img.shape[0] == img_ref.shape[0] and img.shape[1] == img_ref.shape[1])
    for ix in range(-1,2):
        for iy in range(-1,2):
            if (img[dx+ix][dy+iy] == 0 and img_ref[dx+ix][dy+iy] == 1):
                labeling_im(img,img_ref,dx+ix,dy+iy,im_label)
                #print(dx,dy)
                #print(im_label)
def label_ps(image,image_ref):
    im_height = image.shape[0]
    im_width = image.shape[1]
    assert(im_height == image_ref.shape[0] and im_width == image_ref.shape[1])
    label = 1
    for ih in range(im_height-1):
        for iw in range(im_width-1):
            if (image[ih,iw] == 0 and image_ref[ih,iw] == 1):
                labeling_im(image,image_ref,ih,iw,label)
                label = label + 1
    return label-1
def ps_number_detect(ps_data):
    number_ps = np.zeros((ps_data.shape[0],1))
    for frame in range(ps_data.shape[0]):
        count_im = np.zeros((ps_data.shape[1],ps_data.shape[2]))
        ps_count = label_ps(count_im,ps_data[frame,:,:])
        number_ps[frame,0] = ps_count
        del count_im
    return number_ps
def affine_transform(p_fig,p_ref,p_bip):        
    list20 = np.arange(0,20,1)
    for n in range(20):
        if(p_fig[n,0] < 0):
            it = (np.where(list20 == n))
            list20 = np.delete(list20,it)

    pos_X = p_fig[list20,1]
    pos_Y = p_fig[list20,0]
    pos_x = 89-p_ref[list20,1]
    pos_y = p_ref[list20,0]
    sig_X = np.sum(pos_X)
    sig_Y = np.sum(pos_Y)
    sig_XX = np.dot(pos_X,pos_X)
    sig_YY = np.dot(pos_Y,pos_Y)
    sig_x = np.sum(pos_x)
    sig_y = np.sum(pos_y)
    sig_xx = np.dot(pos_x,pos_x)
    sig_yy = np.dot(pos_y,pos_y)
    sig_xy = np.dot(pos_x,pos_y)
    sig_xX = np.dot(pos_x,pos_X)
    sig_yX = np.dot(pos_y,pos_X)
    sig_xY = np.dot(pos_x,pos_Y)
    sig_yY = np.dot(pos_y,pos_Y)
    sig = len(list20)
    af = np.array([[sig_xx,sig_xy,sig_x],
         [sig_xy,sig_yy,sig_y],
         [sig_x,sig_y,sig]
        ])
    bx = np.array([[sig_xX],[sig_yX],[sig_X]])
    by = np.array([[sig_xY],[sig_yY],[sig_Y]])
    coex = np.dot(np.linalg.inv(af),bx)
    coey = np.dot(np.linalg.inv(af),by)
    af_t = np.array([[coex[0][0],coex[1][0],coex[2][0]],[coey[0][0],coey[1][0],coey[2][0]],[0,0,1]])
    p_af_unip = np.dot(af_t,np.array([89-p_ref[:,1],p_ref[:,0],p_ref[:,2]]))
    p_af_bip = np.dot(af_t,np.array([89-p_bip[:,1],p_bip[:,0],p_bip[:,2]]))
    return [p_af_unip,p_af_bip,af_t]

def ps_trj(phasedata,pvdata,rect = 4,path = None,mp4 = 1,rate = 1,col = "k",extrac = 0):
    extra = set_col_extra()
    #extra2 = 
    psg = []
    if not phasedata.shape == pvdata.shape:
        return
    if not os.path.exists(os.path.join(path,"data")):
        os.makedirs(os.path.join(path,"data"))
    if not os.path.exists(os.path.join(path,"phase")):
        os.makedirs(os.path.join(path,"phase"))
    if extrac == 1:
        if not os.path.exists(os.path.join(path,"extra/phase")):
            os.makedirs(os.path.join(path,"extra/phase"))
    for t in range(phasedata.shape[0]):
        count_im = np.zeros((pvdata.shape[1],pvdata.shape[2]))
        ps_count = label_ps(count_im,pvdata[t,:,:])
        ps = np.ndarray((ps_count*2))
        for n in range(1,ps_count+1):
            psn = np.where(count_im == n)
            ps[(n-1)*2] = np.sum(psn[0])/len(psn[0])
            ps[(n-1)*2+1] = np.sum(psn[1])/len(psn[0])
        psg.append(ps)    
        plt.figure(figsize=(4,4))
        plt.imshow(phasedata[t,:,:],cmap = "jet",vmax = np.pi,vmin=-np.pi)
        plt.set_cmap("jet")
        plt.xticks([])
        plt.yticks([])
        plt.text(0,0,"{0} ms".format(t*rate),va = "top",color=col,fontsize = 20)
        for i in range(ps_count):
            ps_rect = np.array([[ps[i*2]-rect,ps[i*2]-rect,ps[i*2]+rect,ps[i*2]+rect,ps[i*2]-rect],
                                 [ps[i*2+1]-rect,ps[i*2+1]+rect,ps[i*2+1]+rect,ps[i*2+1]-rect,ps[i*2+1]-rect]])
            plt.plot(ps_rect[1],ps_rect[0],"k",linewidth = 1.5)
        #plt.xlim(0,phasedata.shape[1]//2)
        #plt.ylim(phasedata.shape[1]//2-phasedata.shape[1]//256,0)
        plt.xlim(0,phasedata.shape[1])
        plt.ylim(phasedata.shape[1],0)
        plt.savefig(os.path.join(path,"phase/{0:0>6}.png".format(t)))
        if extrac == 1:
            plt.set_cmap(extra)
            plt.savefig(os.path.join(path,"extra/phase/{0:0>6}.png".format(t)))
        plt.close()
    fo = open(os.path.join(path,'data/ps_traject.csv'), 'w')
    writer = csv.writer(fo,dialect = csv.excel_tab, lineterminator='\n')
    writer.writerows(psg)
    fo.close()
    if mp4 == 1:   
        mrate = 80*phasedata.shape[0]//2000
        cmd = 'ffmpeg -r {0} -y -i "{1}/phase/%06d.png" -c:v libx264 -pix_fmt yuv420p -qscale 0 "{1}/phase{0}.mp4"'.format(mrate,path)
        print(cmd)
        os.system(cmd)
        if extrac == 1:
            cmd = 'ffmpeg -r {0} -y -i "{1}/extra/phase/%06d.png" -c:v libx264 -pix_fmt yuv420p -qscale 0 "{1}/extra/phase{0}.mp4"'.format(mrate,path)
            print(cmd)
            os.system(cmd)
    return
    