import cv2
import math
from ultralytics import YOLO
import argparse

from config import CLASSES

class DxDetectVideo:
    K = 2
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.5
    THICKNESS = 2
    COLOR_GREEN = (0, 255, 0)
    COLOR_YELLOW = (0, 255, 255) # BGR


    def __init__(self, model_path, in_stream, out_tream):
        self.model = YOLO(model_path)  # load a custom trained model
        self.stream_in = cv2.VideoCapture(in_stream)
        self.WH, self.FPS, self.CC4 = (int(self.stream_in.get(3)//self.K), int(self.stream_in.get(4)//self.K)), \
                                       int(self.stream_in.get(cv2.CAP_PROP_FPS)//self.K), \
                                       cv2.VideoWriter_fourcc(*"XVID")
        self.streame_out = cv2.VideoWriter(out_tream, self.CC4, self.FPS, self.WH)


    def process(self):
        while self.stream_in.isOpened():
            success, frame = self.stream_in.read()
            if not success:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            results = self.model.track(frame, persist=True, stream=True, verbose=False)

            for r in results:
                for box in r.boxes:
                    confidence = math.ceil((box.conf[0] * 100)) / 100
                    if confidence < 0.1:
                        continue

                    id = int(box.id.cpu().numpy()[0]) if box.id else None
                    cls = int(box.cls)

                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    if id is None:
                        cv2.putText(frame, f'?-{CLASSES[cls]}', (x1, y1-10), self.FONT, self.FONT_SCALE, self.COLOR_YELLOW, 2)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), self.COLOR_YELLOW, 3)
                    else:
                        cv2.putText(frame, f'{id}-{CLASSES[cls]}', (x1, y1-10), self.FONT, self.FONT_SCALE, self.COLOR_GREEN, 2)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), self.COLOR_GREEN, 3)
                # end for
            # end for

            cv2.imshow('Video Track', frame)
            self.streame_out.write(cv2.resize(frame, self.WH))

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        # end while

        self.close()


    def close(self):
        # Release the video capture object and close the display window
        self.stream_in.release()
        self.streame_out.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process a video file.')
    parser.add_argument('model_path', type=str, help='Model path')
    parser.add_argument('video_in', type=str, help='Stream in path')
    parser.add_argument('video_out', type=str, help='Stream out path')
    args = parser.parse_args()

    detector = DxDetectVideo(args.model_path, args.video_in, args.video_out)
    detector.process()
