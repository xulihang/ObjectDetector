package org.xulihang;

import java.io.IOException;
import java.util.List;

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
        darknetTest();
        onnxTest();
	}
	
	static private void darknetTest() {
		try {
			DarkNetDetector detector = new DarkNetDetector("C:\\Users\\admin\\Desktop\\git\\ImageTrans\\Objects\\气泡检测模型\\model.cfg","C:\\Users\\admin\\Desktop\\git\\ImageTrans\\Objects\\气泡检测模型\\model.weights",640,960);
			Mat img = Imgcodecs.imread("G:\\imagetrans_project\\sq_test\\2.jpg");
			List<Rect2d> results = detector.Detect(img);
			System.out.println(results);
			System.out.println(results.size());
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	static private void onnxTest() {
		try {
			ONNXDetector detector = new ONNXDetector("G:\\imagetrans_project\\sq_test\\model.onnx");
			 //ObjectDetector detector = new ObjectDetector("G://yolo//yolov8n.onnx",640,640);
	        detector.setWidth(640);
	        detector.setHeight(640);
        	Mat img = Imgcodecs.imread("G:\\imagetrans_project\\sq_test\\2.jpg");
			List<DetectedObject> results = detector.DetectWithClassID(img);
			//System.out.println(results.get(0).classId);
			System.out.println(results);
			System.out.println(results.size());
			for (DetectedObject result:results) {
				Rect2d rect = result.box;
				Imgproc.rectangle(img, new Rect((int)rect.x,(int)rect.y,(int)rect.width,(int)rect.height), new Scalar(255,0,0),5);
		    }
			Imgcodecs.imwrite("G:\\imagetrans_project\\sq_test\\out.jpg",img);
		} catch (Exception e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
       
	}
}
