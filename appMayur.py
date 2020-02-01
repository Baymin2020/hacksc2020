import logging
import time
import edgeiq
"""
Use pose estimation to determine human poses in realtime. Human Pose returns
a list of key points indicating joints that can be used for applications such
as activity recognition and augmented reality.

Pose estimation is only supported using the edgeIQ container with an NCS
accelerator.
"""

#Object Tracking

def main():
    pose_estimator = edgeiq.PoseEstimation("alwaysai/human-pose")
    pose_estimator.load(
            engine=edgeiq.Engine.DNN_OPENVINO,
            accelerator=edgeiq.Accelerator.MYRIAD)

    #instantiate ObjectDetection object
    obj_detect = edgeiq.ObjectDetection("alwaysai/mobilenet_ssd")
    #load engine and accelerator using the load() function
    obj_detect = load(engine = edgeiq.ENGINE.DNN)

    fps = edgeiq.FPS()

    try:
        with edgeiq.WebcamVideoStream(cam=0) as video_stream, \
                edgeiq.Streamer() as streamer:
            # Allow Webcam to warm up
            time.sleep(2.0)
            fps.start()

            # loop detection
            while True:
                frame = video_stream.read()
                results = pose_estimator.estimate(frame)
                # Generate text to display on streamer
                text = ["Model: {}".format(pose_estimator.model_id)]
                text.append(
                        "Inference time: {:1.3f} s".format(results.duration))
                for ind, pose in enumerate(results.poses):
                    text.append("Person {}".format(ind))
                    text.append('-'*10)
                    text.append("Key Points:")
                    for key_point in pose.key_points:
                        text.append(str(key_point))
                streamer.send_data(results.draw_poses(frame), text)

                fps.update()

                if streamer.check_exit():
                    break
    finally:
        fps.stop()

        #Across Frames

        #instantiate CorrelationTracker Object
        tracker = edgeiq.CorrelationTracker()

        #perform object detection and return an object containing predictions
        results = obj_detect.detect_objects(frame)

        for predictions in results.predictions:
            tracker.start(frame, prediction)

        tracker_predictions = tracker.update(frame)

        new_image = edgeiq.markup_image(image, object_detection_predictions)

        #perform another detection
        results = obj_detect.detect_objects(frame)

        #terminate previous tracking ans start tracking new detect_objects
        if tracker.count()
            tracker.stop_all()
        for prediction in results.predictions:
            tracker.start(frame, prediction.box, prediction.label)

        #Across detections
        #obj_detect = edgeiq.ObjectDetection("alwaysai/mobilenet_ssd")
        #load engine and accelerator using the load() function
        #obj_detect = load(engine = edgeiq.ENGINE.DNN)

        #tracker = edgeiq.CentroidTracker()

        #results = obj_detect.detect_objects(frame)

        #objects = tracker.update(results.predictions)

if __name__ == "__main__":
    main()
