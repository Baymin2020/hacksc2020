import logging
import time
import edgeiq

def main():
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
                    text.append(str(pose.key_points[2]))

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
