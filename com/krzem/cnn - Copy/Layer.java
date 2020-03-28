package com.krzem.cnn - Copy;



import java.util.List;



public interface Layer{
	int calc_o(int i);



	int calc_ow(int iw);



	int calc_oh(int ih);



	List<ImagePlate> out(List<ImagePlate> i);



	List<ImagePlate> err(List<ImagePlate> e,double lr);
}