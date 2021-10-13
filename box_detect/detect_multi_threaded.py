from utils import detector_utils
import cv2
import tensorflow as tf
from multiprocessing import Queue, Pool
from utils.detector_utils import WebcamVideoStream
import argparse

frame_processed = 0
score_thresh = 0.21


# 병렬처리 콜백함수
# graph : 점과 선으로 이루어진 수학적 구조? 연산을 표현 session상에서 실행.
# session은 graph의 작업을 GPU나 CPU에 배정하고 실행을 위한 메서드 제공.
# 보통 딥러닝에서는 학습시키기 위한 graph를 만들어서 session을 이용해 학습을 반복 실행 한다.
# 우린 이미 학습된 데이터인 graph를 불러옴!
def worker(input_q, output_q, cap_params, frame_processed):
    print(">> loading frozen model for worker")
    detection_graph, sess = detector_utils.load_inference_graph()
    sess = tf.Session(graph=detection_graph)
    while True:
        frame = input_q.get()
        if (frame is not None):
            boxes, scores = detector_utils.detect_objects(frame, detection_graph, sess)

            # Score가 score_thresh를 넘어서는 박스를 그린다.
            # 박스 위치 반환 받고 싶은데 반환 인자 넣으면 동작이 안돼요 ㅠㅡㅜㅠ
            detector_utils.draw_box_on_image(
                cap_params['num_hands_detect'], cap_params["score_thresh"],
                scores, boxes, cap_params['im_width'], cap_params['im_height'],
                frame)

            output_q.put(frame)
            frame_processed += 1
        else:
            output_q.put(frame)
    sess.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # 카메라 번호
    parser.add_argument(
        '-src',
        '--source',
        dest='video_source',
        type=int,
        default=1,
        help='Device index of the camera.')
    # 찾아낼 손의 최대 수
    parser.add_argument(
        '-nhands',
        '--num_hands',
        dest='num_hands',
        type=int,
        default=1,
        help='Max number of hands to detect.')
    # 화면 너비
    parser.add_argument(
        '-wd',
        '--width',
        dest='width',
        type=int,
        default=300,
        help='Width of the frames in the video stream.')
    # 화면 높이
    parser.add_argument(
        '-ht',
        '--height',
        dest='height',
        type=int,
        default=200,
        help='Height of the frames in the video stream.')
    # 병렬 처리 객체수
    parser.add_argument(
        '-num-w',
        '--num-workers',
        dest='num_workers',
        type=int,
        default=4,
        help='Number of workers.')
    # FIFO 자료형 사이즈
    parser.add_argument(
        '-q-size',
        '--queue-size',
        dest='queue_size',
        type=int,
        default=5,
        help='Size of the queue.')
    args = parser.parse_args()

    # FIFO:선입선출
    input_q = Queue(maxsize=args.queue_size)
    output_q = Queue(maxsize=args.queue_size)

    # video 매핑
    video_capture = WebcamVideoStream(
        src=args.video_source, width=args.width, height=args.height).start()

    # cap_params class에 데이터 입력
    cap_params = {}
    frame_processed = 0
    cap_params['im_width'], cap_params['im_height'] = video_capture.size()
    cap_params['score_thresh'] = score_thresh
    cap_params['num_hands_detect'] = args.num_hands

    print(cap_params, args)


    # 병렬처리 선언 worker : call back routine
    # pool (작업자 프로세스 수, 작업자 프로세스 시작시 호출 함수, 전달인자)
    pool = Pool(args.num_workers, worker,
                (input_q, output_q, cap_params, frame_processed))

    index = 0

    cv2.namedWindow('Multi-Threaded Detection', cv2.WINDOW_NORMAL)

    while True:
        frame = video_capture.read()
        frame = cv2.flip(frame, 1)
        index += 1

        input_q.put(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        output_frame = output_q.get()

        output_frame = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)

        if (output_frame is not None):
            cv2.imshow('Multi-Threaded Detection', output_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("video end")
            break

    pool.terminate()
    video_capture.stop()
    cv2.destroyAllWindows()
