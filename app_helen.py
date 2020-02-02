import logging
import time
import edgeiq

# Human pose detection from local image

"""
Use pose estimation to determine human poses in realtime. Human Pose returns
a list of key points indicating joints that can be used for applications such
as activity recognition and augmented reality.

Pose estimation is only supported using the edgeIQ container with an NCS
accelerator.
"""

def main():
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

            while True:
                frame = video_stream.read()
                results = pose_estimator.estimate(frame)

                text = ["Model: {}".format(pose_estimator.model_id)]

                # Get points
                for ind, pose in enumerate(results.poses):
                    rightShoulder_y = pose.key_points[2][1]
                    leftShoulder_y = pose.key_points[5][1]
                    rightWrist_y = pose.key_points[4][1]
                    leftWrist_y = pose.key_points[7][1]
                    rightHip_y = post.key_points[8][1]
                    leftHip_y = post.key_points[11][1]

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
