package org.xulihang;

import java.io.IOException;

public class Test {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
        System.load("G://opencv_java490.dll");
        ONNXDetector detector = new ONNXDetector("G://yolo//yolov8n.onnx");
        //ObjectDetector detector = new ObjectDetector("G://yolo//yolov8n.onnx",640,640);
        try {
			var results = detector.Detect("G://git//object-detection-and-barcode-reading//IMG20240326163255.jpg");
			System.out.println(results);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

}
