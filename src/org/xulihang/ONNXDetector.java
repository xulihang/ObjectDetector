package org.xulihang;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Rect2d;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.imgcodecs.Imgcodecs;


public class ONNXDetector {
	private Net net;
	private float confThreshold = 0.25f;
	private float nmsThresh = 0.45f;
	private Scalar mean = new Scalar(0,0,0);
	private boolean swapRB = true;
	private boolean crop = false;
	public ONNXDetector(String onnxPath) {
		net = Dnn.readNetFromONNX(onnxPath);
	}
	
	public List<Rect2d> Detect(String imgPath) throws IOException {
		Mat img = Imgcodecs.imread(imgPath);
	    return Detect(img);
	}

	public List<Rect2d> Detect(Mat img) {
		
		Mat blob = Dnn.blobFromImage(img, 1/255, new Size(640,640), mean, swapRB, crop);
	    //Sets the input to the network
	    net.setInput(blob);
	    
	    //Runs the forward pass to get output of the output layers
	    List<Mat> outputs = new ArrayList<>();
    	// Get output layer names
        List<String> outNames = new ArrayList<>();
        List<Integer> outLayers = net.getUnconnectedOutLayers().toList();
        List<String> layersNames = net.getLayerNames();
        for (int i = 0; i < outLayers.size(); ++i)
            outNames.add(layersNames.get(outLayers.get(i)-1));
	    net.forward(outputs,outNames);
	    int rows = (int) outputs.get(0).size().height;
	    int dimensions  = (int) outputs.get(0).size().width;
	    System.out.println(rows);
	    System.out.println(dimensions);
	    if (dimensions > rows) {
	    	rows = (int) outputs.get(0).size().width;
	        dimensions = (int) outputs.get(0).size().height;
	        outputs.set(0,outputs.get(0).reshape(1, dimensions));
	        Core.transpose(outputs.get(0), outputs.get(0));
	    }
	    float[] data = new float[rows * dimensions];
        outputs.get(0).get(0, 0, data);
        float x_factor = img.cols() / 640;
        float y_factor = img.rows() / 640;
        List<Float> confidences = new ArrayList<>();
        List<Rect2d> boxes = new ArrayList<>();
        
        System.out.println("data length");
        System.out.println(data.length);
        for (int i = 0; i < data.length/4; ++i) {
        	
            //double[] classes_scores = getSlice(raw_classes_scores,5,raw_classes_scores.length);
			//Mat scores = new Mat(1, classes_scores.length, CvType.CV_32FC1);
            //scores.put(0, 0, classes_scores);
            //Core.MinMaxLocResult mm = Core.minMaxLoc(scores);
            //double confidence = mm.maxVal;
            //System.out.println("confidence");
            //System.out.println(confidence);
        	float x = data[0+i*4];
            float y = data[1+i*4];
            float w = data[2+i*4];
            float h = data[3+i*4];
            
            int left = (int) ((x - 0.5 * w) * x_factor);
            int top = (int) ((y - 0.5 * h) * y_factor);
            System.out.println(left);
            int width = (int) (w * x_factor);
            int height = (int) (h * y_factor);
            Rect2d box = new Rect2d(left, top, width, height);
            //System.out.println(box);
            //System.out.println(box.width);
            //System.out.println(box.height);
            boxes.add(box);
        }
        //System.out.println(boxes.size());
	    return boxes;
	}
	
	public double[] getSlice(double[] arr, int stIndx, int enIndx) {
		double[] sclicedArr = new double[enIndx - stIndx];

	    for (int i = 0; i < sclicedArr.length; i++) {
	      sclicedArr[i] = arr[stIndx + i];
	    }

	    return sclicedArr;
	  }
}
