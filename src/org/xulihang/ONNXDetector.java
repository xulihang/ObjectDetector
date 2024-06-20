package org.xulihang;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfRect2d;
import org.opencv.core.Rect2d;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.core.Core.MinMaxLocResult;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.imgcodecs.Imgcodecs;


public class ONNXDetector {
	private Net net;
	private int inpHeight = 640;
	private int inpWidth = 640;
	private float confThreshold;
	private float nmsThresh;
	private double scalefactor;
	private Scalar mean;
	private boolean swapRB;
	private boolean crop;
	public ONNXDetector(String onnxPath) throws IOException {
		MatOfByte bytes = ModelUtils.readFileAsMatOfByte(onnxPath);
		net = Dnn.readNetFromONNX(bytes);
		confThreshold=0.3f;
		nmsThresh = 0.5f;
		scalefactor = 1/255.0;
		mean = new Scalar(127.5,127.5,127.5);
		swapRB = true;
		crop = false;
	}
	
	public void setWidth(int width) {
		inpWidth = width;
	}
	
	public void setHeight(int height) {
		inpHeight = height;
	}
	
	public void setConfThreshold(float value) {
		confThreshold=value;
	}

	public void setNMSThreshold(float value) {
		nmsThresh=value;
	}

	public void setScalefactor(double value) {
		scalefactor=value;
	}

	public void setMean(Scalar value) {
		mean=value;
	}

	public void setSwapRB(boolean value) {
		swapRB=value;
	}

	public void setCrop(boolean value) {
		crop=value;
	}
	
	public List<Rect2d> Detect(String imgPath) throws IOException {
		Mat img = Imgcodecs.imread(imgPath);
	    return Detect(img);
	}
	
	public List<Rect2d> Detect(Mat img) {
		List<Rect2d> boxes = new ArrayList<Rect2d>();
		List<DetectedObject> detectedObjects = DetectWithClassID(img);
		for (int i = 0; i < detectedObjects.size(); i++) {
			DetectedObject detectedObject = detectedObjects.get(i);
		    boxes.add(detectedObject.box);
		}
		return boxes;
	}

	public List<DetectedObject> DetectWithClassID(Mat img) {
		
		Mat blob = Dnn.blobFromImage(img, scalefactor, new Size(inpWidth,inpHeight));
		net.setInput(blob);
		Mat predict = net.forward();
		Mat mask = predict.reshape(0,1).reshape(0, predict.size(1));
		
		double width = img.cols() / (double) inpWidth;
		double height = img.rows() / (double) inpHeight;
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
		Dnn.NMSBoxes(bboxes, scores, confThreshold,nmsThresh, indices);
		List<Integer> result = indices.toList();
		List<DetectedObject> detectedObjects = new ArrayList<DetectedObject>();
		for (Integer integer : result) {
			Rect2d rect = new Rect2d(rect2d[integer].tl(),rect2d[integer].size());
			DetectedObject detectedObject = new DetectedObject(rect,classid[integer]);
			detectedObjects.add(detectedObject);
		}
		return detectedObjects;
	}
	
}
