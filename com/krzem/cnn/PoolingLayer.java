package com.krzem.cnn;



import java.lang.Math;
import java.util.ArrayList;
import java.util.List;



public class PoolingLayer implements Layer{
	public int w;
	public int h;
	private boolean[][][] lM;



	public PoolingLayer(int w,int h){
		this.w=w;
		this.h=h;
	}



	@Override
	public int calc_o(int i){
		return i;
	}



	@Override
	public int calc_ow(int iw){
		return iw/this.w+(iw%this.w>0?1:0);
	}



	@Override
	public int calc_oh(int ih){
		return ih/this.h+(ih%this.h>0?1:0);
	}



	@Override
	public double[][][] out(double[][][] in){
		if (this.lM==null){
			this.lM=new boolean[in.length][in[0].length][in[0][0].length];
		}
		double[][][] o=new double[in.length][this.calc_oh(in[0].length)][this.calc_ow(in[0][0].length)];
		for (int i=0;i<in.length;i++){
			for (int j=0;j<this.calc_oh(in[0].length);j++){
				for (int k=0;k<this.calc_ow(in[0][0].length);k++){
					double mv=Double.MIN_VALUE;
					int ml=-1;
					int mm=-1;
					for (int l=Math.min(j*this.h,in[i].length-1);l<Math.min(Math.min(j*this.h,in[i].length-1)+this.h,in[i].length);l++){
						for (int m=Math.min(k*this.w,in[i][0].length-1);m<Math.min(Math.min(k*this.w,in[i][0].length-1)+this.w,in[i][0].length);m++){
							double v=in[i][l][m];
							if (v>mv){
								mv=v;
								ml=l;
								mm=m;
							}
						}
					}
					this.lM[i][ml][mm]=true;
					o[i][j][k]=mv;
				}
			}
		}
		return o;
	}



	@Override
	public double[][][] err(double[][][] g,double lr){
		double[][][] o=new double[g.length][this.lM[0].length][this.lM[0][0].length];
		for (int i=0;i<g.length;i++){
			for (int j=0;j<this.lM[0].length;j++){
				for (int k=0;k<this.lM[0][0].length;k++){
					if (this.lM[i][j][k]==true){
						o[i][j][k]=g[i][j/this.h][k/this.w];
					}
				}
			}
		}
		return o;
	}
}