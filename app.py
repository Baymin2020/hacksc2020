import logging
import time
import edgeiq
# Including if-statements

def main():
    rightShoulder_y = 0
    leftShoulder_y = 0
    rightElbow_y = 0
    leftElbow_y = 0
    rightWrist_y = 0
    leftWrist_y = 0

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

                for ind, pose in enumerate(results.poses):
                    leftShoulder_y = pose.key_points[5][1]

                for ind, pose in enumerate(results.poses):
                    rightElbow_y = pose.key_points[3][1]

                for ind, pose in enumerate(results.poses):
                    leftElbow_y = pose.key_points[6][1]

                for ind, pose in enumerate(results.poses):
                    rightWrist_y = pose.key_points[4][1]

                for ind, pose in enumerate(results.poses):
                    leftWrist_y = pose.key_points[7][1]

                text.append(str(rightShoulder_y))
                text.append(str(leftShoulder_y))
                text.append(str(rightElbow_y))
                text.append(str(leftElbow_y))
                text.append(str(rightWrist_y))
                text.append(str(leftWrist_y))

                # if rightElbow_y > rightShoulder_y and rightWrist_y > rightShoulder_y:


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
