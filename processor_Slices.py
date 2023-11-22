# https://www.ccoderun.ca/programming/doxygen/opencv_3.2.0/tutorial_py_watershed.html
# https://pyimagesearch.com/2015/11/02/watershed-opencv/#download-the-code
# 10c: 16.3 mm
# 25c: 20.8 mm
# 50c: 23 mm
# 1 UAH: 26 mm
# --------------------------------------------------------------------------------------------------------------------
import os
import numpy
import cv2
import time
import pandas as pd
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings("ignore")
# --------------------------------------------------------------------------------------------------------------------
import tools_DF
import tools_image
import tools_draw_numpy
import tools_time_profiler
import tools_optical_flow
import tools_plot_v2
import tools_plot_dancing
# --------------------------------------------------------------------------------------------------------------------
class processor_Slices(object):
    def __init__(self,folder_out,pix_per_mm=None):
        self.name = "detector_slices"
        self.folder_out = folder_out
        self.frame_id =0
        self.pix_per_mm = pix_per_mm
        self.roi_pad = 20
        self.tol_track_delta = 10
        self.diametr_min = 30
        self.diametr_max = 130
        self.dict_granules_mm = {16.3: 'XS',18: 'S', 20.8: 'M',23: 'L',26: 'XL' }
        self.colors_size_mm = tools_draw_numpy.get_colors(2 * len(self.dict_granules_mm.keys()), colormap='gist_rainbow')[:len(self.dict_granules_mm.keys())][::-1]
        self.df_stats = pd.DataFrame([])
        self.filename_stats = 'df.csv'

        self.T = tools_time_profiler.Time_Profiler(verbose=False)
        self.OF = tools_optical_flow.OpticalFlow_LucasKanade(None,folder_out=folder_out)
        self.P = tools_plot_v2.Plotter(self.folder_out)
        self.PD = tools_plot_dancing.Plotter_dancing(folder_out)

        self.image_dashboard = numpy.full((720,1280,3),255,dtype=numpy.uint8)
        self.image_dashboard = tools_image.put_image(self.image_dashboard,cv2.imread('./images/header_1280.png'),0,0)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def reset_stats(self, image_start):
        self.frame_id = 0
        self.start_time = time.time()
        self.OF.init_start_frame(image_start)
        self.df_stats = pd.DataFrame([])
        self.df_histo_historic = pd.DataFrame([])

        for l,c in zip(self.dict_granules_mm.values(),self.colors_size_mm):
            self.PD.P.set_color(l, c)

        if os.path.isfile(self.folder_out + self.filename_stats):
            os.remove(self.folder_out + self.filename_stats)
        if image_start is not None:
            self.ellipses_prev = None
            self.ellipses_id_prev = None
            self.obj_count = 0
            self.H,self.W = image_start.shape[:2]
        return
# ----------------------------------------------------------------------------------------------------------------------
    def binarize(self, gray):
        #binarized = cv2.threshold(cv2.cvtColor(cv2.pyrMeanShiftFiltering(255-gray, 21, 51), cv2.COLOR_BGR2GRAY), 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        #binarized  = cv2.threshold(255-gray, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        binarized = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        # blockSize = 27
        # maxValue = 255
        # binarized = cv2.adaptiveThreshold(gray, maxValue, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize,0)
        return binarized
# ----------------------------------------------------------------------------------------------------------------------
    def append_stats(self,tags_sizes):

        df = pd.DataFrame({'frameID':self.frame_id,
                           'objID':[i for i in self.ellipses_id_cur],
                           'x':[e[0][0] for e in self.ellipses_cur],
                           'y':[e[0][1] for e in self.ellipses_cur],
                           'rx_px':[e[1][0] for e in self.ellipses_cur],
                           'ry_px':[e[1][1] for e in self.ellipses_cur],
                           'rx_mm':[e[1][0]/self.pix_per_mm for e in self.ellipses_cur],
                           'ry_mm':[e[1][1]/self.pix_per_mm for e in self.ellipses_cur],
                           'tag':[t.split('(')[1][:-1] for t in tags_sizes]})

        self.df_stats = pd.concat([self.df_stats,df])
        # if not os.path.isfile(self.folder_out+self.filename_stats):
        #     df.to_csv(self.folder_out+self.filename_stats,index=False,float_format='%.2f')
        # else:
        #     df.to_csv(self.folder_out+self.filename_stats, index=False,float_format='%.2f',mode='a', header=False)

        return
# ----------------------------------------------------------------------------------------------------------------------
    def build_histo_current(self,figsize):
        # df_agg = tools_DF.my_agg(self.df_stats,cols_groupby=['objID','tag'],cols_value=['frameID'],aggs=['count'],list_res_names=['appearances'])
        # df_agg = df_agg[df_agg['appearances']>=3]
        # df_agg2 = tools_DF.my_agg(df_agg, cols_groupby=['tag'], cols_value=['objID'], aggs=['count'])
        # image = None

        df_cur = self.df_stats[self.df_stats['frameID'] == self.frame_id]
        df_cur_agg = tools_DF.my_agg(df_cur, cols_groupby=['tag'], cols_value=['frameID'], aggs=['count'],list_res_names=['#'])
        df_chart = pd.DataFrame({'size':list(self.dict_granules_mm.values())})
        df_chart = tools_DF.fetch(df_chart,'size',df_cur_agg,'tag','#')
        df_chart.fillna(0, inplace=True)
        values = df_chart.iloc[:, 1].values
        labels = df_chart.iloc[:, 0].values
        fig = self.P.plot_bars(values,labels,colors=self.colors_size_mm[:,[2,1,0]]/255.0,figsize=figsize,filename_out=None)
        image = self.P.get_image(fig, clr_bg=(0.95,0.95,0.95))

        return image,values,labels
# ----------------------------------------------------------------------------------------------------------------------
    def build_histo_historic(self, values,labels,figsize):

        df = pd.DataFrame({'tag':labels,'frameID':self.frame_id,'#':values})
        self.df_histo_historic = pd.concat([self.df_histo_historic,df])

        image = numpy.full((int(100 * figsize[1]), int(100 * figsize[0]), 3), 255, dtype=numpy.uint8)
        if numpy.unique(self.df_histo_historic.iloc[:,1]).shape[0]>3:
            image = self.PD.plot_stacked_data(self.df_histo_historic[self.df_histo_historic.iloc[:,1]>self.frame_id-20], idx_time=1, idx_label=0, idx_value=2, top_objects=6,out_format_x='%d',major_step=None,alpha=0.8,figsize=figsize)

        return image
# ----------------------------------------------------------------------------------------------------------------------
    def get_segments(self,image_bin):
        D = ndimage.distance_transform_edt(image_bin)
        localMax = peak_local_max(D, indices=False, min_distance=20, labels=image_bin)

        markers = ndimage.label(localMax, structure=numpy.ones((3, 3)))[0]
        image_segments = watershed(-D, markers, mask=image_bin)
        return image_segments.astype(numpy.uint8)
# ----------------------------------------------------------------------------------------------------------------------
    def get_contours(self, image_segments):
        contours = []
        for label in numpy.unique(image_segments):
            if label == 0: continue
            mask = numpy.zeros(image_segments.shape[:2], dtype="uint8")
            mask[image_segments == label] = 255
            for c in cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
                contours.append(c)

        return contours
# ----------------------------------------------------------------------------------------------------------------------
    def is_good_ellipse(self,e):

        result = True
        if not(e[0][0] > self.roi_pad and e[0][0] < self.W - self.roi_pad and e[0][1] > self.roi_pad and e[0][1] < self.H - self.roi_pad):
            result = False

        ratio = float(e[1][0]/(e[1][1]+1e-4))
        if ratio>1.25 or 1/(ratio+1e-4)>1.25:
            result = False

        diametr  = (e[1][0] +e[1][1])/2
        if diametr < self.diametr_min or diametr>self.diametr_max:
            result = False

        return result
# ----------------------------------------------------------------------------------------------------------------------
    def get_ellipses(self, image_segments):
        ellipses = []
        for label in numpy.unique(image_segments):
            if label == 0: continue
            mask = numpy.zeros(image_segments.shape[:2], dtype="uint8")
            mask[image_segments == label] = 255
            contour = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            points = numpy.array(contour).reshape((-1, 2))
            if points.shape[0]<10:
                continue
            e = cv2.fitEllipse(points)  # ( (cx,cy),(rx,ry),a)

            if self.is_good_ellipse(e):
                ellipses.append(e)

        return ellipses
# ----------------------------------------------------------------------------------------------------------------------
    def get_colors_by_size_mm(self,ellipses):
        colors = []
        for ellipse in ellipses:
            axes = (int(ellipse[1][0] / 2), int(ellipse[1][1] / 2))
            diametr_pix = (axes[0]+axes[1])
            diametr_mm = diametr_pix/self.pix_per_mm

            d = [abs(diametr_mm - d) for d in self.dict_granules_mm.keys()]
            colors.append(list(self.colors_size_mm)[numpy.argmin(d)])

        return colors
# ----------------------------------------------------------------------------------------------------------------------
    def get_labels_by_size_mm(self,ellipses):
        labels = []
        for ellipse in ellipses:
            axes = (int(ellipse[1][0] / 2), int(ellipse[1][1] / 2))
            diametr_pix = (axes[0] + axes[1])
            diametr_mm = diametr_pix / self.pix_per_mm

            d = [abs(diametr_mm - d) for d in self.dict_granules_mm.keys()]
            label = list(self.dict_granules_mm.values())[numpy.argmin(d)]
            labels.append('%.1f mm (%s)'%(diametr_mm,label))

        return labels
# ----------------------------------------------------------------------------------------------------------------------
    def do_tracking(self,M):
        if (self.ellipses_id_prev is None) or (M is None):
            ids = [self.obj_count+i for i in range(len(self.ellipses_cur))]
            self.obj_count+=len(self.ellipses_cur)
        else:
            xy_cur  = numpy.array([e[0] for e in self.ellipses_cur])
            if xy_cur.shape[0]==0:
                return []
            xy_prev = numpy.array([e[0] for e in self.ellipses_prev]).reshape((-1, 1, 2))
            if xy_prev.shape[0]==0:
                ids = [i+self.obj_count for i in range(xy_cur.shape[0])]
                self.obj_count = 1+max(ids)
                return ids

            xy_cur_cand = cv2.perspectiveTransform(xy_prev, M).reshape((-1, 2))

            distance_matrix = cdist(xy_cur, xy_cur_cand, 'euclidean')
            idxs  = numpy.argmin(distance_matrix, axis=1)
            dsts = [distance_matrix[row,idx] for row,idx in enumerate(idxs)]
            ids0 = [(self.ellipses_id_prev[idx] if dst<self.tol_track_delta else numpy.nan) for idx,dst in zip(idxs,dsts)]
            ids = []
            for i in ids0:
                if numpy.isnan(i):
                    ids.append(int(self.obj_count))
                    self.obj_count+=1
                else:
                    ids.append(i)

        return ids
# ----------------------------------------------------------------------------------------------------------------------
    def draw_fps(self,image):
        if time.time() > self.start_time:
            fps = self.frame_id / (time.time() - self.start_time)
            image = tools_draw_numpy.draw_text_fast(image,'%.1f fps | %d'%(fps,self.frame_id),(self.W - 95, self.H - 15), (0, 0, 0))
        return image
# ----------------------------------------------------------------------------------------------------------------------
    def next_step(self):
        self.ellipses_prev = self.ellipses_cur
        self.ellipses_id_prev = self.ellipses_id_cur
        self.frame_id += 1
        return
# ----------------------------------------------------------------------------------------------------------------------
    def compose_dashboard(self,image_camera,image_histo_historic,image_histo_current):

        self.image_dashboard = tools_image.put_image(self.image_dashboard,image_camera       , 170, 53)
        self.image_dashboard = tools_image.put_image(self.image_dashboard,image_histo_historic,170, 750)
        self.image_dashboard = tools_image.put_image(self.image_dashboard,image_histo_current, 410, 750)

        shift = 8

        self.image_dashboard = tools_draw_numpy.draw_rect_fast(self.image_dashboard,  53-shift, 170-shift,  53+self.W-shift, 170+self.H+shift, (0,0,0), w=1)
        H, W = image_histo_historic.shape[:2]
        self.image_dashboard = tools_draw_numpy.draw_rect_fast(self.image_dashboard, 750-shift, 170-shift, 750+W-shift     , 170+H-shift, (0,0,0),w=1)
        H, W = image_histo_current.shape[:2]
        self.image_dashboard = tools_draw_numpy.draw_rect_fast(self.image_dashboard, 750 - shift, 410 + shift,750 + W - shift, 410 + H + shift, (0, 0, 0), w=1)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def process_file_contours(self, filename_in,do_debug=True):

        image = cv2.imread(filename_in) if isinstance(filename_in,str) else filename_in

        self.T.tic('gray')
        gray = tools_image.bgr2hsv(image)[:,:,1]  #day mode
        #gray  = tools_image.bgr2hsv(image)[:,:,0]  #night mode

        self.T.tic('bin')
        image_bin = self.binarize(gray)
        self.T.tic('get_segments')
        image_segments = self.get_segments(image_bin)
        self.T.tic('get_ellipses')
        self.ellipses_cur = self.get_ellipses(image_segments)
        self.T.tic('evaluate_flow')
        M = self.OF.evaluate_flow(image)
        self.T.tic('do_tracking')
        self.ellipses_id_cur = self.do_tracking(M)
        self.T.tic('get_colors_by_size_mm')
        colors = self.get_colors_by_size_mm(self.ellipses_cur)
        self.T.tic('get_labels_by_size_mm')
        tags_sizes = self.get_labels_by_size_mm(self.ellipses_cur)
        labels = ['ID:'+str(l_id)+' '+l_size for l_id,l_size in zip(self.ellipses_id_cur,tags_sizes)]
        xy = numpy.array([( e[0][0], e[0][1]-e[1][1]/2) for e in self.ellipses_cur])

        self.T.tic('draw_ellipses')
        image_camera = tools_draw_numpy.draw_ellipses(tools_image.desaturate(image), self.ellipses_cur, color=colors,w=2,transperency=0.60)
        self.T.tic('draw_texts')
        image_camera = tools_draw_numpy.draw_texts(image_camera, labels, xy, clrs_fg=(0,0,0),clrs_bg=colors,font_size=14)
        self.T.tic('draw_fps')
        image_camera = self.draw_fps(image_camera)
        self.T.tic('append_stats')
        self.append_stats(tags_sizes)
        self.T.tic('build_histo_current')
        image_histo_current,values,labels = self.build_histo_current((4.70, 2.40))
        self.T.tic('build_histo_historic')
        image_histo_historic = self.build_histo_historic(values,labels,(4.70, 2.40))


        self.compose_dashboard(image_camera,image_histo_historic,image_histo_current)

        if do_debug:
            cv2.imwrite(self.folder_out + '1gray.png', gray)
            cv2.imwrite(self.folder_out + '2image_bin.png', image_bin)
            cv2.imwrite(self.folder_out + '3image_segments.png', image_segments)


        self.T.tic('next step')
        self.OF.next_step()
        self.next_step()

        return self.image_dashboard
# ----------------------------------------------------------------------------------------------------------------------

