package org.xulihang;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;

import org.opencv.core.MatOfByte;

public class ModelUtils {
	public static MatOfByte readFileAsMatOfByte(String path) throws IOException {
		File file = new File(path);
		byte[] bytes = Files.readAllBytes(file.toPath());
		MatOfByte mat = new MatOfByte(bytes);
		return mat;
	}
}
