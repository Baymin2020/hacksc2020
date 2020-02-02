import logging
import time
import edgeiq
# Including if-statements

def main():
    # The current frame index
    frame_idx = 0
    # The number of frames to skip before running detector
    detect_period = 30

    obj_detect = edgeiq.ObjectDetection(
        "alwaysai/ssd_mobilenet_v1_coco_2018_01_28")
    obj_detect.load(engine=edgeiq.Engine.DNN)

    tracker = edgeiq.CorrelationTracker(max_objects=5)

    rightShoulder_y = 0
    leftShoulder_y = 0

    rightWrist_y = 0
    leftWrist_y = 0

    rightHip_y = 0
    leftHip_y = 0

    pose_estimator = edgeiq.PoseEstimation("alwaysai/human-pose")
    pose_estimator.load(
            engine=edgeiq.Engine.DNN_OPENVINO,
            accelerator=edgeiq.Accelerator.MYRIAD)

    fps = edgeiq.FPS()

    try:
        with edgeiq.WebcamVideoStream(cam=0) as video_stream, \
                edgeiq.Streamer() as streamer:
            time.sleep(2.0)
            fps.start()

            startTime = time.time()
            futureTime = startTime + 20

            while True:
                frame = video_stream.read()
                results = pose_estimator.estimate(frame)

                text = ["Model: {}".format(pose_estimator.model_id)]

                # Get right shoulder points
                for ind, pose in enumerate(results.poses):
                    rightShoulder_y = pose.key_points[2][1]
                    leftShoulder_y = pose.key_points[5][1]
                    rightWrist_y = pose.key_points[4][1]
                    leftWrist_y = pose.key_points[7][1]
                    rightHip_y = pose.key_points[8][1]
                    leftHip_y = pose.key_points[11][1]

                if (rightWrist_y < rightShoulder_y) or (leftWrist_y < leftShoulder_y):
                    text.append("Mood: Happy")
                elif (rightWrist_y < rightHip_y) and (leftWrist_y < leftHip_y):
                    text.append("Mood: Angry")
                elif ((rightWrist_y >= rightHip_y + 20) or (rightWrist_y <= rightHip_y - 20)) and ((leftWrist_y >= leftHip_y + 20) or (leftWrist_y <= leftHip_y - 20)):
                    text.append("Mood: Idle")
                else:
                    text.append("Mood: Idle")

                detectionResults = obj_detect.detect_objects(frame, confidence_level=.5)

                if tracker.count:
                    tracker.stop_all()

                predictions = detectionResults.predictions
                boxList = []
                boxNameList = []

                for prediction in predictions:
                    tracker.start(frame, prediction)

                    if prediction.label == 'person':
                        if not boxNameList:
                            boxList.append(prediction.box)
                            boxNameList.append(prediction.label)
                        else:
                            for name in boxNameList:
                                if name == prediction.label:
                                    break
                                else:
                                    boxList.append(prediction.box)
                                    boxNameList.append(prediction.label)
                    elif prediction.label == 'chair':
                        if not boxNameList:
                            boxList.append(prediction.box)
                            boxNameList.append(prediction.label)
                        else:
                            for name in boxNameList:
                                if name == prediction.label:
                                    break
                                else:
                                    boxList.append(prediction.box)
                                    boxNameList.append(prediction.label)

                if len(boxList) >= 2:
                    distance = boxList[0].compute_distance(boxList[1])
                    if abs(distance) < 115:
                        text.append(str("At chair"))
                        if time.time() >= futureTime:
                            text.append(str("Get Out"))
                    else:
                        text.append(str("Not in chair"))
                        futureTime = time.time() + 20

                frame = edgeiq.markup_image(frame, predictions, show_labels=True,
                                            show_confidences=False, colors=obj_detect.colors)

                streamer.send_data(results.draw_poses(frame), text)

                streamer.send_data(frame, text)

                frame_idx += 1

                fps.update()

                if streamer.check_exit():
                    break
    finally:
        tracker.stop_all()
        fps.stop()
        print("elapsed time: {:.2f}".format(fps.get_elapsed_seconds()))
        print("approx. FPS: {:.2f}".format(fps.compute_fps()))

        print("Program Ending")


if __name__ == "__main__":
    main()
