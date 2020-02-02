import logging
import time
import edgeiq
# Including if-statements

def main():
    rightShoulder_y = 0
    leftShoulder_y = 0

    # rightElbow_y = 0
    # leftElbow_y = 0

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

            while True:
                frame = video_stream.read()
                results = pose_estimator.estimate(frame)

                text = ["Model: {}".format(pose_estimator.model_id)]

                # Get right shoulder points
                for ind, pose in enumerate(results.poses):
                    text.append(str("Right Shoulder Y"))
                    rightShoulder_y = pose.key_points[2][1]
                    text.append(str(rightShoulder_y))

                    text.append(str("Left Shoulder Y"))
                    leftShoulder_y = pose.key_points[5][1]
                    text.append(str(leftShoulder_y))

                    # text.append(str("Right Elbow Y"))
                    # rightElbow_y = pose.key_points[3][1]
                    # text.append(str(rightElbow_y))

                    # text.append(str("Left Elbow Y"))
                    # leftElbow_y = pose.key_points[6][1]
                    # text.append(str(leftElbow_y))

                    text.append(str("Right Wrist Y"))
                    rightWrist_y = pose.key_points[4][1]
                    text.append(str(rightWrist_y))

                    text.append(str("Left Wrist Y"))
                    leftWrist_y = pose.key_points[7][1]
                    text.append(str(leftWrist_y))

                    text.append(str("Right Hip Y"))
                    rightHip_y = pose.key_points[8][1]
                    text.append(str(rightHip_y))

                    text.append(str("Left Hip Y"))
                    leftHip_y = pose.key_points[11][1]
                    text.append(str(leftHip_y))

                if (rightWrist_y < rightShoulder_y) or (leftWrist_y < leftShoulder_y):
                    text.append("Mood: Happy")
                elif (rightWrist_y < rightHip_y) and  (leftWrist_y < leftHip_y):
                    text.append("Mood: Angry")
                else:
                    text.append("Mood: Idle")

                streamer.send_data(results.draw_poses(frame), text)

                fps.update()

                if streamer.check_exit():
                    break
    finally:
        fps.stop()
        print("elapsed time: {:.2f}".format(fps.get_elapsed_seconds()))
        print("approx. FPS: {:.2f}".format(fps.compute_fps()))

        print("Program Ending")


if __name__ == "__main__":
    main()
