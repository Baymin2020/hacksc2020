import logging
import time
import edgeiq
# Including if-statements

def main():
    rightShoulder_y = 0
    leftShoulder_y = 0

    rightElbow_y = 0
    leftElbow_y = 0

    rightElbow_x = 0
    leftElbow_x = 0

    rightWrist_y = 0
    rightWrist_x = 0

    leftWrist_y = 0
    leftWrist_x = 0

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
                    rightShoulder_y = pose.key_points[2][1]
                    leftShoulder_y = pose.key_points[5][1]
                    rightElbow_y = pose.key_points[3][1]
                    rightElbow_x = pose.key_points[3][0]
                    leftElbow_y = pose.key_points[6][1]
                    leftElbow_x = pose.key_points[6][0]
                    rightWrist_y = pose.key_points[4][1]
                    rightWrist_x = pose.key_points[4][0]
                    leftWrist_y = pose.key_points[7][1]
                    leftWrist_x = pose.key_points[7][0]

                # text.append(str(rightShoulder_y))
                # text.append(str(leftShoulder_y))
                # text.append(str(rightElbow_y))
                # text.append(str(leftElbow_y))
                # text.append(str(rightWrist_y))
                # text.append(str(leftWrist_y))

                if rightWrist_y > rightShoulder_y and rightElbow_y > rightShoulder_y and leftWrist_y > leftShoulder_y and leftElbow_y > leftShoulder_y:
                    text.append("Mood: Happy")
                elif abs(rightWrist_y - leftElbow_y) < 5 and abs(rightWrist_x - leftElbow_x) < 5 and abs(lightWrist_y - rightElbow_y) < 5 and abs(leftWrist_x - rightElbow_x) < 5:
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
