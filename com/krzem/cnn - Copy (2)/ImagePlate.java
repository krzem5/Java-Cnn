package com.krzem.cnn - Copy (2);



public class ImagePlate{
	public int w;
	public int h;
	private double[][] dt;



	public ImagePlate(double[][] dt){
		this.set(dt);
	}



	public int size(){
		return this.w*this.h;
	}



	public void set(double[][] ndt){
		this.dt=ndt;
		this._store_size();
	}



	public double get(int x,int y){
		return this.dt[x][y];
	}



	public ImagePlate convolve(ImagePlate m){
		double[][] o=new double[this.h-m.h+1][this.w-m.w+1];
		for (int i=0;i<this.h-m.h+1;i++){
			for (int j=0;j<this.w-m.w+1;j++){
				double sum=0;
				for (int k=0;k<m.h;k++){
					for (int l=0;l<m.w;l++){
						int x=i+k;
						int y=j+l;
						if (x<0||x>=this.h||y<0||y>=this.w){
							continue;
						}
						sum+=m.get(k,l)*this.get(x,y);
					}
				}
				o[i][j]=sum;
			}
		}
		return new ImagePlate(o);
	}



	public ImagePlate activation(String af){
		double[][] o=new double[this.h][this.w];
		for (int i=0;i<this.h;i++){
			for (int j=0;j<this.w;j++){
				o[i][j]=Activation.apply(this.get(i,j),af);
			}
		}
		return new ImagePlate(o);
	}



	public double[] array(){
		double[] o=new double[this.size()];
		int off=0;
		for (int i=0;i<this.h;i++){
			for (int j=0;j<this.w;j++){
				o[off]=this.get(i,j);
				off++;
			}
		}
		return o;
	}



	private void _store_size(){
		this.w=this.dt[0].length;
		this.h=this.dt.length;
	}



public class ImagePlate{
	public int w;
	public int h;
	private double[][] dt;



	public ImagePlate(double[][] dt){
		this.set(dt);
	}



	public int size(){
		return this.w*this.h;
	}



	public void set(double[][] ndt){
		this.dt=ndt;
		this._store_size();
	}



	public double get(int x,int y){
		return this.dt[x][y];
	}



	public ImagePlate convolve(ImagePlate m){
		double[][] o=new double[this.h-m.h+1][this.w-m.w+1];
		for (int i=0;i<this.h-m.h+1;i++){
			for (int j=0;j<this.w-m.w+1;j++){
				double sum=0;
				for (int k=0;k<m.h;k++){
					for (int l=0;l<m.w;l++){
						int x=i+k;
						int y=j+l;
						if (x<0||x>=this.h||y<0||y>=this.w){
							continue;
						}
						sum+=m.get(k,l)*this.get(x,y);
					}
				}
				o[i][j]=sum;
			}
		}
		return new ImagePlate(o);
	}



	public ImagePlate activation(String af){
		double[][] o=new double[this.h][this.w];
		for (int i=0;i<this.h;i++){
			for (int j=0;j<this.w;j++){
				o[i][j]=Activation.apply(this.get(i,j),af);
			}
		}
		return new ImagePlate(o);
	}



	public double[] array(){
		double[] o=new double[this.size()];
		int off=0;
		for (int i=0;i<this.h;i++){
			for (int j=0;j<this.w;j++){
				o[off]=this.get(i,j);
				off++;
			}
		}
		return o;
	}



	private void _store_size(){
		this.w=this.dt[0].length;
		this.h=this.dt.length;
	}
}