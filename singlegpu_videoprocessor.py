import cv2
import os
import numpy as np
from datetime import datetime
from typing import Optional

from kalman_track import Sort
from logger import setup_logger
from config import Config

class ParallelVideoProcessor:
    def __init__(self, model_manager):
        self.logger = setup_logger('parallel_video_processor', 'parallel_video_processor.log')
        self.logger.info("Initializing ParallelVideoProcessor.")

        self.model_manager = model_manager
        
        self.MAX_AGE = Config.MAX_AGE
        self.MIN_MA = Config.MIN_MA
        
        # Create frames output directory
        self.frames_base_dir = Config.FRAMES_OUTPUT_DIR
        os.makedirs(self.frames_base_dir, exist_ok=True)

    def initialize_trackers(self):
        """Initialize separate trackers for each class"""
        trackers = {}
        for class_id in range(4):  # Assuming 4 classes (0-3)
            trackers[class_id] = Sort(self.MAX_AGE, self.MIN_MA)
        return trackers

    async def process_video(self, video_path: str, video_type: str):
        """Process video using available model instance"""
        try:
            # Get available model instance
            model_instance = await self.model_manager.get_available_model(video_type)
            self.logger.info(f"Got {video_type} model instance.")

            try:
                success = await self._process_hotspot_video(video_path, model_instance)
                return success
            finally:
                # Always release the model instance back to the pool
                await self.model_manager.release_model(model_instance)

        except Exception as e:
            self.logger.error(f"Error processing video: {str(e)}", exc_info=True)
            raise
    
    async def process_frame_hotspot(self, frame, model_instance, conf_threshold=0.50):
        """Process frame for hotspot detection using available model instance"""
        try:
            results = model_instance.model(frame, conf=conf_threshold)
            # Get bounding boxes for detected objects
            boxes = []
            detections = []
            
            for r in results:
                boxes_tensor = r.boxes.xyxy.cpu()   # Get boxes in xyxy format
                confs = r.boxes.conf.cpu()          # Get confidence scores
                cls = r.boxes.cls.cpu()             # Get class indices
                
                for box, conf, cl in zip(boxes_tensor, confs, cls):
                    if conf >= conf_threshold:
                        x1, y1, x2, y2 = map(int, box[:4])
                        cx = (x1 + x2) / 2.
                        cy = (y1 + y2) / 2.
                        ar = (x2 - x1) / (y2 - y1)
                        h = (y2 - y1)
                        # if cx <= 100 or cx >= 220:
                        #     continue
                        detections.append({
                            'bbox': [cx, cy, ar, h],
                            'confidence': float(conf),
                            'class': int(cl)
                        })
            
            return results, detections, boxes

        except Exception as e:
            self.logger.error(f"Error processing hotspot frame: {str(e)}", exc_info=True)
            raise

    async def _process_hotspot_video(self, video_path: str, model_instance):
        """Process hotspot video with given model instance"""
        self.logger.info(f"Starting hotspot detection for video_id")

        try:
            # Initialize separate trackers for each class
            self.logger.info("Initializing Kalman trackers for each class")
            trackers = self.initialize_trackers()

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise Exception(f"Failed to open video file: {video_path}")

            frame_count = 0
            frame_processed = 0
            tracked_objects = {class_id: {} for class_id in range(4)}  # Track objects per class
            processing_start = datetime.now()
            self.logger.info("Starting frame processing loop")

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                if frame_count % 10 == 0:  # Process every 5th frame
                    frame_processed += 1

                    # Process frame
                    results, detections, _ = await self.process_frame_hotspot(
                        frame, model_instance
                    )
                    # Organize detections by class
                    class_detections = {class_id: [] for class_id in range(4)}
                    for det in detections:
                        class_id = det['class']
                        bbox = det['bbox']
                        class_detections[class_id].append(bbox)

                    # Update trackers for each class
                    annotated_frame = results[0].plot()
                    
                    for class_id, dets in class_detections.items():
                        if len(dets) > 0:
                            dets_array = np.array(dets)
                            tracked_bbox, count = trackers[class_id].update(dets_array)
                        else:
                            tracked_bbox, count = trackers[class_id].update()

                        # Update tracked objects for this class
                        if len(tracked_bbox) > 0:
                            for track in tracked_bbox:
                                object_id = int(track[-1])
                                centroid = (int(track[0]), int(track[1]))
                                
                                if object_id not in tracked_objects[class_id]:
                                    tracked_objects[class_id][object_id] = {
                                        'frames_tracked': 0,
                                        'last_position': centroid,
                                        'class_id': class_id
                                    }
                                
                                tracked_objects[class_id][object_id]['frames_tracked'] += 1
                                tracked_objects[class_id][object_id]['last_position'] = centroid

                                # Draw tracking info on frame
                                cv2.putText(annotated_frame, f"Class {class_id} ID {object_id}", 
                                            (centroid[0] - 10, centroid[1] - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                cv2.circle(annotated_frame, centroid, 4, (0, 255, 0), -1)

                    # Save frame periodically (every 30 processed frames)
                    if frame_processed % 10 == 0:
                        frame_filename = f"frame_{frame_count}.jpg"
                        temp_frame_path = os.path.join(self.frames_base_dir, frame_filename)
                        
                        try:
                            cv2.imwrite(temp_frame_path, annotated_frame)
                            self.logger.debug(f"Frame saved: {temp_frame_path}")
                            
                        except Exception as e:
                            self.logger.error(f"Failed to save frame: {str(e)}")

            cap.release()
            self.logger.info(f"Video processing completed. Total frames: {frame_processed}")
            
            # Store final results
            processing_time = (datetime.now()-processing_start).total_seconds()
            total_tracked_objects = sum(len(objs) for objs in tracked_objects.values())
            
            final_results = {
                'total_frames': frame_processed,
                'unique_tracked_objects': total_tracked_objects,
                'objects_per_class': {
                    class_id: len(objs) 
                    for class_id, objs in tracked_objects.items()
                }
            }
            print(final_results)
            
            self.logger.info(f"Processing completed in {processing_time:.2f} seconds. "
                        f"Found {total_tracked_objects} unique objects across all classes")
            
            return True

        except Exception as e:
            self.logger.error(f"Hotspot video processing failed: {str(e)}", exc_info=True)
            raise

    def cleanup(self):
        """Cleanup resources"""
        self.db_writer.shutdown()