import os
import argparse
import numpy
import cv2
# ----------------------------------------------------------------------------------------------------------------------
import sys
# ----------------------------------------------------------------------------------------------------------------------
sys.path.insert(1, './tools/')
import processor_Slices
import configurations
import tools_draw_numpy
import tools_video
from CV import tools_calibrate
from CV import tools_pr_geom
# ----------------------------------------------------------------------------------------------------------------------
folder_out = './images/output/'
# ---------------------------------------------------------------------------------------------------------------------
def transform_image(image,homography_3x3,target_W,target_H):
    if homography_3x3 is not None:
        image = cv2.warpPerspective(image, homography_3x3, (target_W, target_H))
        image = image[::-1,:]
    #image = tools_image.transpone_image(image)
    return image
# ---------------------------------------------------------------------------------------------------------------------
def calibrate_cam(target_W=640,target_H=480):
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    cap.set(3, target_W)
    cap.set(4, target_H)
    image = cap.read()[1]

    leg = target_H * 0.50
    board_mm = 74.0
    x = numpy.linspace(target_W/2-leg/2,target_W/2+leg/2, 7)
    y = numpy.linspace(target_H/2-leg/2,target_H/2+leg/2, 7)
    targets_2d = numpy.array(numpy.meshgrid(x,y)).T.astype(numpy.float32).reshape((-1,2))
    H = None

    while (True):
        points_2d = tools_calibrate.get_chessboard_corners(image, chess_rows=7, chess_cols=7)
        if points_2d is not None and points_2d.shape[0]==7*7:
            points_2d = points_2d.astype(numpy.float32)
            image = tools_draw_numpy.draw_points(image, points_2d.reshape((-1,2)))
            H,result = tools_pr_geom.fit_homography(points_2d.reshape((-1,1,2)),targets_2d.reshape((-1,1,2)),method = cv2.RANSAC,do_debug=False)
            image = cv2.warpPerspective(image, H, (target_W, target_H))
            points_2d_proj = cv2.perspectiveTransform(points_2d.reshape((-1,1,2)),H).reshape((-1,2))
            image = tools_draw_numpy.draw_points(image, points_2d_proj.reshape((-1, 2)),color=(0,100,200))

        cv2.imshow('Calibration', image)
        key = cv2.waitKey(1)
        image = cap.read()[1]
        if key & 0xFF == 27: break
        if key == 13 or key==32:
            print('[[%1.5f,%1.5f,%1.5f],\n[%1.5f,%1.5f,%1.5f],\n[%1.5f,%1.5f,%1.5f]]'%tuple(H.flatten()))
            print('pix per mm')
            lx_px = (numpy.max(points_2d_proj[:,0])-numpy.min(points_2d_proj[:,0]))
            ly_px = (numpy.max(points_2d_proj[:,1])-numpy.min(points_2d_proj[:,1]))
            print('H pix per mm:',lx_px/(board_mm*6/8))
            print('V pix per mm:',ly_px/(board_mm*6/8))

    cv2.destroyAllWindows()
    cap.release()
    return
# ----------------------------------------------------------------------------------------------------------------------
def process_loop(P,filename_in=None,use_camera=False,camera_W=640,camera_H=480,homography_3x3=None):
    if use_camera:
        cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        cap.set(3, camera_W)
        cap.set(4, camera_H)
        image = cap.read()[1]
    else:
        if 'mp4' in filename_in:
            cap = cv2.VideoCapture(filename_in)
            image = cap.read()[1]
        else:
            cap = None
            image = cv2.imread(filename_in) if isinstance(filename_in, str) else filename_in
            image = cv2.resize(image,(camera_W, camera_H))

    image = transform_image(image, homography_3x3, camera_W, camera_H)
    P.reset_stats(image)

    while(True):

        result = P.process_file_contours(image)
        cv2.imshow('Granulometrics', result)

        key = cv2.waitKey(1)

        if use_camera or ('mp4' in filename_in):
            image = cap.read()[1]
            if image is None: break
            image = transform_image(image,homography_3x3, camera_W, camera_H)

        if key & 0xFF == 27: break
        if image is None: break
        if key == 13 or key == 32:cv2.imwrite(folder_out+'screenshot.png',image)
        if key in [ord('r'),ord('R')]:P.reset_stats(image)

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    P.T.stage_stats(folder_out + 'time.csv')
    return
# ----------------------------------------------------------------------------------------------------------------------
#tools_video.capture_video(1, folder_out + 'ex_08.mp4')
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    print(sys.argv)
    print(' '.join([k for k in os.environ.keys()]))
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('--mode', '-m', help='mode of granulometrics\nstatic: process static image\noffline: process offline video\nlive: process live video feed from cam', default='static')
    args = parser.parse_args()
    if args.mode =='static':
        config = configurations.dct_config_static_image
    elif args.mode =='offline':
        config = configurations.dct_config_video_offline
    elif args.mode =='live':
        config = configurations.dct_config_video_live
    elif args.mode == 'calibration':
        calibrate_cam()
        exit(1)
    else:
        parser.print_help()
        exit(1)

    # P = processor_Slices.processor_Slices(folder_out, pix_per_mm=config['pix_per_mm'])
    # process_loop(P,filename_in=config['filename_in'],use_camera=config['use_camera'],homography_3x3=config['homography_3x3'])

