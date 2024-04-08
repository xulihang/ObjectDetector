package org.xulihang;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfRect;
import org.opencv.core.Rect2d;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.DetectionModel;
import org.opencv.imgcodecs.Imgcodecs;


public class ObjectDetector {

	private DetectionModel net;
	private int inpHeight;
	private int inpWidth;
	private float confThreshold;
	private float nmsThresh;
	private double scalefactor;
	private Scalar mean;
	private boolean swapRB;
	private boolean crop;

	public ObjectDetector(String modelPath,int width,int height) {
		net = new DetectionModel(modelPath);
		inpWidth=width;
		inpHeight=height;
		init();
	}

	public ObjectDetector(String modelPath, String modelCfg, int width,int height) {
		net = new DetectionModel(modelPath,modelCfg);
		inpWidth=width;
		inpHeight=height;
		init();
	}

	public void init() {
	    confThreshold=0.2f;
		nmsThresh = 0.5f;
		scalefactor = 1/127.5;
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
		net.setInputSize(new Size(inpWidth,inpHeight));
		Scalar scaleFactorScalar = new Scalar(scalefactor);
		net.setInputScale(scaleFactorScalar);
		net.setInputMean(mean);
		net.setInputSwapRB(swapRB);
		net.setInputCrop(crop);
		MatOfInt classIds = new MatOfInt();
		MatOfFloat confidences = new MatOfFloat();
		MatOfRect rects = new MatOfRect();
	    net.detect(img, classIds, confidences, rects,confThreshold,nmsThresh);
	    List<Rect2d> rect2ds = new ArrayList<>();
	    for (int i=0;i<rects.rows();i++){
		    //double classId=classIds.get(i,0)[0];
		    double confidence=confidences.get(i,0)[0];
		    if (confidence>0.5){
		    	int left,top,width,height;
		    	left=(int) rects.get(i,0)[0];
		    	top=(int) rects.get(i,0)[1];
		    	width=(int) rects.get(i,0)[2];
		    	height=(int) rects.get(i,0)[3];
		    	Rect2d rect = new Rect2d(left,top,width,height);
		    	rect2ds.add(rect);
		    }
		    System.out.println();
	    }

	    return rect2ds;
	}


}

