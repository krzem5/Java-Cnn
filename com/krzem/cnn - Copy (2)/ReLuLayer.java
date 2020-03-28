package com.krzem.cnn - Copy (2);



import java.lang.Math;



public class ReLu implements Layer{
	public int calc_o(int i){
		return i;
	}



	public int calc_ow(int iw){
		return iw;
	}



	public int calc_oh(int ih){
		return ih;
	}



	public double apply(double i,){
		return Math.max(i,0.01);
	}



	public double applyD(double i){
		return (i>0.01?1:0);
	}



	public List<ImagePlate> out(List<ImagePlate> i){
		List<ImagePlate> ol=new ArrayList<ImagePlate>();
		for (ImagePlate p:i){
			double[][] o=new double[p.h][p.w];
			for (int i=0;i<p.h;i++){
				for (int j=0;j<p.w;j++){
					o[i][j]=Activation.apply(p.get(i,j),af);
				}
			}
			ol.add(new ImagePlate(o));
		}
		return ol;
	}



	public List<ImagePlate> err(List<ImagePlate> e,double lr);
}