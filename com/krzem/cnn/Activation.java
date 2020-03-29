package com.krzem.cnn;



import java.lang.Math;



public class Activation{
	public static double apply(double i,String f){
		if (f.equals("sigmoid")){
			return 1d/(1+Math.exp(-i));
		}
		else if (f.equals("relu")){
			return Math.max(i,0.01);
		}
		return i;
	}



	public static double applyD(double i,String f){
		if (f.equals("sigmoid")){
			return i*(1-i);
		}
		else if (f.equals("relu")){
			return (i>0.01?1:0);
		}
		return i;
	}
}