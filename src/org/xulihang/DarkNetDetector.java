package org.xulihang;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.List;


import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfRect2d;
import org.opencv.core.MatOfByte;  
import org.opencv.core.Point;
import org.opencv.core.Rect2d;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.imgcodecs.Imgcodecs;

import org.opencv.utils.Converters;


public class DarkNetDetector {

	private Net net;
	private int inpHeight;
	private int inpWidth;
	private float confThreshold;
	private float nmsThresh;
	private double scalefactor;
	private Scalar mean;
	private boolean swapRB;
	private boolean crop;

	public DarkNetDetector(Net loadedNet,int width,int height) {
		net=loadedNet;
		inpWidth=width;
		inpHeight=height;
		init();
	}

	public DarkNetDetector(String darkNetConfig, String modelWeights, int width,int height) throws IOException {
        try {
    		net = Dnn.readNetFromDarknet(darkNetConfig, modelWeights);	
        }catch (Exception e) {
        	MatOfByte configMat = ModelUtils.readFileAsMatOfByte(darkNetConfig);
            MatOfByte weightsMat = ModelUtils.readFileAsMatOfByte(modelWeights);
    		net = Dnn.readNetFromDarknet(configMat, weightsMat);
        }
		inpWidth=width;
		inpHeight=height;
		init();
	}

	public void init() {
	    net.setPreferableBackend(Dnn.DNN_BACKEND_OPENCV);
	    net.setPreferableTarget(Dnn.DNN_TARGET_CPU);
	    confThreshold=0.2f;
		nmsThresh = 0.5f;
		scalefactor = 1/255.0;
		mean = new Scalar(0,0,0);
		swapRB = false;
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
	    //Create a 4D blob from a frame.
	    Mat blob = Dnn.blobFromImage(img, scalefactor, new Size(inpWidth,inpHeight), mean, swapRB, crop);
	    //Sets the input to the network
	    net.setInput(blob);
	    //Runs the forward pass to get output of the output layers
	    List<Mat> outs = new ArrayList<>();
        List<String> outBlobNames = getOutputNames();
	    net.forward(outs,outBlobNames);
	    return postprocess(img,outs,true);
	}

	//Get the names of the output layers
	private List<String> getOutputNames() {
	    //Get the names of all the layers in the network
	    List<String> layersNames = net.getLayerNames();
	    MatOfInt m=net.getUnconnectedOutLayers();
	    List<String> names= new ArrayList<String>();
	    for (int i=0;i<m.rows();i++) {
	    	int index = (int) m.get(i, 0)[0]-1;
	    	names.add(layersNames.get(index));
	    }
	    //Get the names of the output layers, i.e. the layers with unconnected outputs
	    return names;
	}

	private List<Rect2d> postprocess(Mat img, List<Mat> result,Boolean nms) {
		int imgWidth=img.cols();
		int imgHeight=img.rows();
        List<Integer> clsIds = new ArrayList<>();
        List<Float> confs = new ArrayList<>();
        List<Rect2d> rects2d = new ArrayList<>();
        int num=0;
        for (int i = 0; i < result.size(); ++i)
        {
            // each row is a candidate detection, the 1st 4 numbers are
            // [center_x, center_y, width, height], followed by (N-4) class probabilities
            Mat level = result.get(i);
            for (int j = 0; j < level.rows(); ++j)
            {
                Mat row = level.row(j);
                Mat scores = row.colRange(5, level.cols());
                Core.MinMaxLocResult mm = Core.minMaxLoc(scores);
                float confidence = (float)mm.maxVal;
                Point classIdPoint = mm.maxLoc;
                if (confidence > confThreshold)
                {
                	num=num+1;
                    int centerX = (int)(row.get(0,0)[0] * imgWidth); //scaling for drawing the bounding boxes//
                    int centerY = (int)(row.get(0,1)[0] * imgHeight);
                    int width   = (int)(row.get(0,2)[0] * imgWidth);
                    int height  = (int)(row.get(0,3)[0] * imgHeight);
                    int left    = centerX - width  / 2;
                    int top     = centerY - height / 2;

                    clsIds.add((int)classIdPoint.x);
                    confs.add((float)confidence);
                    rects2d.add(new Rect2d(left,top,width,height));
                }
            }
        }
        if (nms==true && rects2d.size()>1) {

	        Mat converted = Converters.vector_float_to_Mat(confs);
	        MatOfFloat confidences = new MatOfFloat(converted);
	        Rect2d[] boxesArray = rects2d.toArray(new Rect2d[0]);
	        MatOfRect2d boxes = new MatOfRect2d(boxesArray);
	        //boxes.fromArray(boxesArray);
	        MatOfInt indices = new MatOfInt();
	        Dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThresh, indices);
	        int [] ind = indices.toArray();
	        List<Rect2d> rectsAfterNMS = new ArrayList<>();
	        for (int i = 0; i < ind.length; ++i)
	        {
	            int idx = ind[i];
	            Rect2d box = boxesArray[idx];
	            int width   = (int)box.width;
                int height  = (int)box.height;
                int left    = (int)box.x;
                int top     = (int)box.y;
	            rectsAfterNMS.add(new Rect2d(left,top,width,height));
	        }
	        return rectsAfterNMS;
        }else {
        	return rects2d;
        }
	}
}

