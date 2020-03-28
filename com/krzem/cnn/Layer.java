package com.krzem.cnn;



import java.util.List;



public interface Layer{
	public int calc_o(int i);



	public int calc_ow(int iw);



	public int calc_oh(int ih);



	public double[][][] out(double[][][] i);



	public double[][][] err(double[][][] e,double lr);
}