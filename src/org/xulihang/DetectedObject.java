package org.xulihang;

import org.opencv.core.Rect2d;

public class DetectedObject {

	public Rect2d box;
	public int classId;
	public float theta = 0;
	public DetectedObject(Rect2d box, int classId) {
		this.box = box;
		this.classId = classId;
	}
}
