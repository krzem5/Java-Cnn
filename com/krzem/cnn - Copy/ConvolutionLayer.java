package com.krzem.cnn - Copy;



import java.util.ArrayList;
import java.util.List;



public class ConvolutionLayer implements Layer{
	public int w;
	public int h;
	public int d;
	public int n;
	private List<List<ImagePlate>> convL;
	private List<ImagePlate> lI;
	private List<ImagePlate> lO;



	public ConvolutionLayer(int w,int h,int d,int n){
		this.w=w;
		this.h=h;
		this.d=d;
		this.n=n;
		this._gen_conv();
	}
	


	@Override
	public int calc_o(int i){
		return i/this.d;
	}



	@Override
	public int calc_ow(int iw){
		return iw-this.w+1;
	}



	@Override
	public int calc_oh(int ih){
		return ih-this.h+1;
	}



	@Override
	public List<ImagePlate> out(List<ImagePlate> in){
		this.lI=in;
		List<ImagePlate> o=new ArrayList<ImagePlate>();
		for (int i=0;i<this.n;i++){
			double[][] dt=new double[in.get(0).h-this.h+1][in.get(0).w-this.w+1];
			for (int j=0;j<this.d;j++){
				ImagePlate conv=in.get(j).convolve(this.convL.get(i).get(j));
				for (int k=0;k<in.get(0).h-this.h+1;k++){
					for (int l=0;l<in.get(0).w-this.w+1;l++){
						dt[k][l]+=conv.get(k,l);
					}
				}
			}
			o.add(new ImagePlate(dt).activation("relu"));
		}
		this.lO=o;
		return o;
	}



	@Override
	public List<ImagePlate> err(List<ImagePlate> e,double lr){
		List<ImagePlate> o=new ArrayList<ImagePlate>();
		double[][][] err=new double[e.size()][][];
		double[][][] dlt=new double[e.size()][][];
		for (int i=0;i<e.size();i++){
			err[i]=new double[this.lI.get(0).h][this.lI.get(0).w];
			dlt[i]=new double[this.lI.get(0).h][this.lI.get(0).w];
			for (int j=0;j<this.d;j++){
				for (int k=0;k<this.lI.get(0).h-this.h;k++){
					for (int l=0;l<this.lI.get(0).w-this.w;l++){
						for (int m=0;m<this.h;m++){
							for (int n=0;n<this.w;n++){
								err[i][k+m][l+n]+=e.get(i).get(k,l)*this.convL.get(i).get(j).get(m,n);
							}
						}
					}
				}
				for (int k=0;k<this.lI.get(0).h;k++){
					for (int l=0;l<this.lI.get(0).w;l++){
						dlt[i][k][l]+=err[i][k][l]*Activation.applyD(this.lI.get(j).get(k,l),"relu");
					}
				}
			}
			o.add(new ImagePlate(dlt[i]));
		}
		return o;
	}



	private void _gen_conv(){
		this.convL=new ArrayList<List<ImagePlate>>();
		for (int i=0;i<this.n;i++){
			List<ImagePlate> cc=new ArrayList<ImagePlate>();
			for (int j=0;j<this.d;j++){
				double[][] dt=new double[this.h][this.w];
				for (int k=0;k<this.h;k++){
					for (int l=0;l<this.w;l++){
						dt[k][l]=Random.nextG();
					}
				}
				cc.add(new ImagePlate(dt));
			}
			this.convL.add(cc);
		}
	}
}