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
        ONNXDetector detector = new ONNXDetector("C:\\Users\\admin\\Desktop\\git\\ImageTrans\\Objects\\model.onnx");
        //ObjectDetector detector = new ObjectDetector("G://yolo//yolov8n.onnx",640,640);
        detector.setWidth(1024);
        detector.setHeight(1024);
        try {
        	Mat img = Imgcodecs.imread("G:\\imagetrans_project\\sq_test\\4d8b0b03ly1g3f1969vkhj20i26skhdv.jpg");
			List<Rect2d> results = detector.Detect(img);
			System.out.println(results);
			System.out.println(results.size());
			for (Rect2d rect:results) {
				Imgproc.rectangle(img, new Rect((int)rect.x,(int)rect.y,(int)rect.width,(int)rect.height), new Scalar(255,0,0),5);
			}
			Imgcodecs.imwrite("G:\\imagetrans_project\\sq_test\\out.jpg",img);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
}
