package com.krzem.cnn;



public class Random{
	public static final int SEED=12345;
	private static java.util.Random RG=new java.util.Random(SEED);



	public static double nextG(){
		return Random.RG.nextGaussian();
	}
}