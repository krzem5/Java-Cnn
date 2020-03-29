package com.krzem.cnn;



import java.util.ArrayList;
import java.util.List;



public class ConvolutionLayer implements Layer{
	private String act;
	private double[][][][] convL;
	private double[][][] lI;



	public ConvolutionLayer(int cn,int d,int kw,int kh,String act){
		this.convL=this._gen_conv(cn,d,kw,kh);
		this.act=act;
	}



	public int convn(){
		return this.convL.length;
	}



	public int convw(){
		return this.convL[0][0][0].length;
	}



	public int convh(){
		return this.convL[0][0].length;
	}



	@Override
	public int calc_o(int i){
		return i/this.convL[0].length;
	}



	@Override
	public int calc_ow(int iw){
		return iw-this.convL[0][0][0].length+1;
	}



	@Override
	public int calc_oh(int ih){
		return ih-this.convL[0][0].length+1;
	}



	@Override
	public double[][][] out(double[][][] in){
		this.lI=in;
		double[][][] o=new double[this.convL.length][this.calc_oh(in[0].length)][this.calc_ow(in[0][0].length)];
		for (int i=0;i<this.convL.length;i++){
			for (int j=0;j<this.convL[0].length;j++){
				for (int k=0;k<this.calc_oh(in[0].length);k++){
					for (int l=0;l<this.calc_ow(in[0][0].length);l++){
						for (int m=0;m<this.convL[i][j].length;m++){
							for (int n=0;n<this.convL[i][j][0].length;n++){
								int x=l+n;
								int y=k+m;
								if (x<0||x>=in[0].length||y<0||y>=in[0][0].length){
									continue;
								}
								o[i][k][l]+=this.convL[i][j][m][n]*in[j][y][x];
							}
						}
						o[i][k][l]=Activation.apply(o[i][k][l]/* + BIAS*/,this.act);
					}
				}
			}
		}
		return o;
	}



	@Override
	public double[][][] err(double[][][] e,double lr){
		double[][][] o=new double[e.length][this.lI[0].length][this.lI[0][0].length];
		double[][][] err=new double[e.length][this.lI[0].length][this.lI[0][0].length];
		for (int i=0;i<e.length;i++){
			for (int j=0;j<this.convL[0].length;j++){
				for (int k=0;k<this.lI[0].length-this.convL[0][0].length;k++){
					for (int l=0;l<this.lI[0][0].length-this.convL[0][0][0].length;l++){
						for (int m=0;m<this.convL[0][0].length;m++){
							for (int n=0;n<this.convL[0][0][0].length;n++){
								err[i][k+m][l+n]+=e[i][k][l]*this.convL[i][j][m][n];
							}
						}
					}
				}
				for (int k=0;k<this.lI[0].length;k++){
					for (int l=0;l<this.lI[0][0].length;l++){
						o[i][k][l]+=err[i][k][l]*Activation.applyD(this.lI[j][k][l],this.act);
					}
				}
			}
		}
		return o;
	}



	private double[][][][] _gen_conv(int cn,int d,int kw,int kh){
		double[][][][] o=new double[cn][d][kh][kw];
		for (int i=0;i<cn;i++){
			for (int j=0;j<d;j++){
				for (int k=0;k<kh;k++){
					for (int l=0;l<kw;l++){
						o[i][j][k][l]=Random.nextG();
					}
				}
			}
		}
		return o;
	}
}