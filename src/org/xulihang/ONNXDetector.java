package org.xulihang;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfRect2d;
import org.opencv.core.Rect2d;
import org.opencv.core.Size;
import org.opencv.core.Core.MinMaxLocResult;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.imgcodecs.Imgcodecs;


public class ONNXDetector {
	private Net net;
	public ONNXDetector(String onnxPath) {
		net = Dnn.readNetFromONNX(onnxPath);
	}
	
	public List<Rect2d> Detect(String imgPath) throws IOException {
		Mat img = Imgcodecs.imread(imgPath);
	    return Detect(img);
	}

	public List<Rect2d> Detect(Mat img) {
		
		Mat blob = Dnn.blobFromImage(img, 1/255.0, new Size(640,640));
		net.setInput(blob);
		Mat predict = net.forward();
		Mat mask = predict.reshape(0,1).reshape(0, predict.size(1));
		
		double width = img.cols() /640.0;
		double height = img.rows() /640.0;
		Rect2d[] rect2d = new Rect2d[mask.cols()];
		float[] scoref = new float[mask.cols()];
		int[] classid = new int[mask.cols()];
		for (int i = 0; i < mask.cols(); i++) {
			double[] x = mask.col(i).get(0, 0);
			double[] y = mask.col(i).get(1, 0);
			double[] w = mask.col(i).get(2, 0);
			double[] h = mask.col(i).get(3, 0);
			rect2d[i] = new Rect2d((x[0]-w[0]/2)*width, (y[0]-h[0]/2)*height, w[0]*width, h[0]*height);
			Mat score = mask.col(i).submat(4, predict.size(1)-1, 0, 1);
			MinMaxLocResult mmr = Core.minMaxLoc(score);
			scoref[i] = (float)mmr.maxVal;
			classid[i] = (int) mmr.maxLoc.y;
		}
		MatOfRect2d bboxes = new MatOfRect2d(rect2d);
		MatOfFloat scores = new MatOfFloat(scoref);
		MatOfInt indices = new MatOfInt();
		Dnn.NMSBoxes(bboxes, scores, 0.3f, 0.5f, indices);
		List<Integer> result = indices.toList();
		List<Rect2d> boxes = new ArrayList<Rect2d>();
		for (Integer integer : result) {
			boxes.add(new Rect2d(rect2d[integer].tl(),rect2d[integer].size()));
		}
		return boxes;
	}
	
}
