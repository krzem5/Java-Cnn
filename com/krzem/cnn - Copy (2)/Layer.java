package com.krzem.cnn - Copy (2);



import java.util.List;



public interface Layer{
	public int calc_o(int i);



	public int calc_ow(int iw);



	public int calc_oh(int ih);



	public List<ImagePlate> out(List<ImagePlate> i);



	public List<ImagePlate> err(List<ImagePlate> e,double lr);
}