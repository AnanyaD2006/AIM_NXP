# Copyright 2025 NXP

# Copyright 2016 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from ultralytics import YOLO
import rclpy
from rclpy.node import Node
from synapse_msgs.msg import WarehouseShelf
from sensor_msgs.msg import CompressedImage
import cv2
import numpy as np

QOS_PROFILE_DEFAULT = 10


class ObjectRecognizer(Node):
	""" Initializes object recognizer node with the required publishers and subscriptions.

		Returns:
			None
	"""
	def __init__(self):
		super().__init__('object_recognizer')

		self.max_objects_per_shelf = 6

		# Subscription for camera images.
		self.subscription_camera = self.create_subscription(
			CompressedImage,
			'/camera/image_raw/compressed',
			self.camera_image_callback,
			QOS_PROFILE_DEFAULT)

		# Publisher for Shelf Objects.
		self.publisher_shelf_objects = self.create_publisher(
			WarehouseShelf,
			'/shelf_objects',
			QOS_PROFILE_DEFAULT)

		# Publisher for output image (for debug purposes).
		self.publisher_object_recog = self.create_publisher(
			CompressedImage,
			"/debug_images/object_recog",
			QOS_PROFILE_DEFAULT)

		# Load YOLOv8 model
		self.model = YOLO('./src/NXP_AIM_INDIA_2025/br3b_ros_aim_india/yolov8n.pt')
		self.label_names = self.model.names

		self.confidence_threshold = 0.65

		self.iou_threshold = 0.5
		self.sharpening_kernel= np.array([
			[0, -1, 0],
			[-1, 5, -1],
			[0, -1, 0]
			], dtype=np.float32)

		


	""" Publishes images for debugging purposes.

		Args:
			publisher: ROS2 publisher of the type sensor_msgs.msg.CompressedImage.
			image: image given by an n-dimensional numpy array.

		Returns:
			None
	"""
	def publish_debug_image(self, publisher, image):
		if image.size:
			message = CompressedImage()
			_, encoded_data = cv2.imencode('.jpg', image)
			message.format = "jpeg"
			message.data = encoded_data.tobytes()
			publisher.publish(message)


	""" Analyzes the image received from /camera/image_raw/compressed to detect shelf objects.
		Publishes the existence of objects in the image on the /shelf_objects topic.

		Args:
			message: "docs.ros.org/en/melodic/api/sensor_msgs/html/msg/CompressedImage.html"

		Returns:
			None
	"""
	def camera_image_callback(self, message):
		# Convert message to an n-dimensional numpy array representation of image.
		np_arr = np.frombuffer(message.data, np.uint8)
		image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
		sharpened_image = cv2.filter2D(image, -1, self.sharpening_kernel)

		results = self.model(sharpened_image, conf=self.confidence_threshold,iou=self.iou_threshold)

		shelf_objects_message = WarehouseShelf()
		object_count_dict = {}
		all_detected_boxes = []

		for result in results:
			for box in result.boxes:
				all_detected_boxes.append((float(box.conf[0]), int(box.cls[0]), box.xyxy[0].cpu().numpy().astype(int)))

		all_detected_boxes.sort(key=lambda x: x[0], reverse=True)
		filtered_boxes = all_detected_boxes[:self.max_objects_per_shelf]

		for conf, cls_id, xyxy in filtered_boxes:
			object_name = self.label_names[cls_id]
			object_count_dict[object_name] = object_count_dict.get(object_name, 0) + 1

			

			cv2.rectangle(sharpened_image, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0,255,0), 2)
			label_text = f"{object_name} {conf:.2f}"
			cv2.putText(sharpened_image, f"{object_name} {conf:.2f}", (xyxy[0], xyxy[1]-10),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)


		for key, value in object_count_dict.items():
			shelf_objects_message.object_name.append(key)
			shelf_objects_message.object_count.append(value)

		self.publisher_shelf_objects.publish(shelf_objects_message)
		self.publish_debug_image(self.publisher_object_recog, sharpened_image)


def main(args=None):
	rclpy.init(args=args)

	object_recognizer = ObjectRecognizer()

	rclpy.spin(object_recognizer)

	# Destroy the node explicitly
	# (optional - otherwise it will be done automatically
	# when the garbage collector destroys the node object)
	object_recognizer.destroy_node()
	rclpy.shutdown()


if __name__ == '__main__':
	main()

