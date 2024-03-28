package org.xulihang;

import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.core.Rect2d;
import org.opencv.core.Scalar;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

public class Test {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
        System.load("G://opencv_java490.dll");
        ONNXDetector detector = new ONNXDetector("G://yolo//yolov8n.onnx");
        //ObjectDetector detector = new ObjectDetector("G://yolo//yolov8n.onnx",640,640);
        try {
        	Mat img = Imgcodecs.imread("G://git//object-detection-and-barcode-reading//IMG20240326163255.jpg");
			var results = detector.Detect(img);
			System.out.println(results);
			System.out.println(results.size());
			for (Rect2d rect:results) {
				Imgproc.rectangle(img, new Rect((int)rect.x,(int)rect.y,(int)rect.width,(int)rect.height), new Scalar(255,0,0),5);
			}
			Imgcodecs.imwrite("G://git//object-detection-and-barcode-reading//out.jpg",img);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
}
