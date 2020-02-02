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
    image_paths = edgeiq.list_images(base_path="C:\Users\hanha\Documents\GitHub\hacksc2020\pic")
    # Replace base_path with local image_path 
    image = cv2.imread()
    pose_estimator = edgeiq.PoseEstimation("alwaysai/human-pose")
    pose_estimator.load(
            engine = edgeiq.Engine.DNN_OPENVINO,
            accelerator = edgeiq.Accelerator.MYRIAD)

    # print("Loaded model:\n{}\n".format(pose_estimator.model_id))
    # print("Engine: {}".format(pose_estimator.engine))

    results = pose_estimator.estimate(image)
    text = ["Model: {}".format(pose_estimator.model_id)]
            for ind, pose in enumerate(results.poses):
                text.append("Person {}".format(ind))
                text.append('-'*10)
                text.append("Key Points:")
                for key_point in pose.key_points:
                    text.append(str(key_point))
                (results.draw_poses(image), text)


if __name__ == "__main__":
    main()
