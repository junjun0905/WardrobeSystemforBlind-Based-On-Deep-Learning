from utils import detector_utils as detector_utils
import cv2
import argparse

detection_graph, sess = detector_utils.load_inference_graph()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-sth',
        '--scorethreshold',
        dest='score_thresh',
        type=float,
        default=0.2,
        help='Score threshold for displaying bounding boxes')
    parser.add_argument(
        '-src',
        '--source',
        dest='video_source',
        default=1,
        help='Device index of the camera.')
    parser.add_argument(
        '-wd',
        '--width',
        dest='width',
        type=int,
        default=320,
        help='Width of the frames in the video stream.')
    parser.add_argument(
        '-ht',
        '--height',
        dest='height',
        type=int,
        default=180,
        help='Height of the frames in the video stream.')
    parser.add_argument(
        '-num-w',
        '--num-workers',
        dest='num_workers',
        type=int,
        default=4,
        help='Number of workers.')
    parser.add_argument(
        '-q-size',
        '--queue-size',
        dest='queue_size',
        type=int,
        default=5,
        help='Size of the queue.')
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video_source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    im_width, im_height = (cap.get(3), cap.get(4))

    num_hands_detect = 2

    cv2.namedWindow('Single-Threaded Detection', cv2.WINDOW_NORMAL)

    while True:

        ret, image_np = cap.read()
 
        try:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        except:
            print("Error converting to RGB")



        boxes, scores = detector_utils.detect_objects(image_np,
                                                      detection_graph, sess)


        detector_utils.draw_box_on_image(num_hands_detect, args.score_thresh,
                                         scores, boxes, im_width, im_height,
                                         image_np)


        cv2.imshow('Single-Threaded Detection',
                   cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

        if cv2.waitKey(5) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

