package com.krzem.cnn - Copy (2);



public class FullyConnectedLayer{
	public int i;
	public int n;
	public String af;
	private double[][] w;
	private double[] lI;
	private double[] lO;



	public FullyConnectedLayer(int i,int n,String af){
		this.i=i;
		this.n=n;
		this.af=af;
		this._gen_w();
	}



	public double[] out(double[] in){
		for (int i=0;i<in.length;i++){
			this.lI[i]=in[i];
		}
		for (int i=0;i<this.n;i++){
			double sum=0;
			for (int j=0;j<this.lI.length;j++){
				sum+=this.w[i][j]*this.lI[j];
			}
			this.lO[i]=Activation.apply(sum,this.af);
		}
		return this.lO;
	}



	public double[] err(double[] dlt,double lr){
		double[] ndlt=new double[this.i];
		for (int i=0;i<this.i;i++){
			int sum=0;
			for (int j=0;j<this.n;j++){
				sum+=dlt[j]*this.w[j][i]*Activation.applyD(this.lI[i],this.af);
			}
			ndlt[i]=sum;
		}
		for (int i=0;i<this.n;i++){
			for (int j=0;j<this.i+1;j++){
				this.w[i][j]-=dlt[i]*this.lI[j]*lr;
			}
		}
		return ndlt;
	}



	private void _gen_w(){
		this.lI=new double[this.i+1];
		this.lI[this.i]=-1;
		this.lO=new double[this.n];
		this.w=w=new double[this.n][this.i+1];
		for (int i=0;i<this.n;i++){
			for (int j=0;j<this.i+1;j++) {
				this.w[i][j]=Math.random();
			}
		}
	}



public class FullyConnectedLayer{
	public int i;
	public int n;
	public String af;
	private double[][] w;
	private double[] lI;
	private double[] lO;



	public FullyConnectedLayer(int i,int n,String af){
		this.i=i;
		this.n=n;
		this.af=af;
		this._gen_w();
	}



	public double[] out(double[] in){
		for (int i=0;i<in.length;i++){
			this.lI[i]=in[i];
		}
		for (int i=0;i<this.n;i++){
			double sum=0;
			for (int j=0;j<this.lI.length;j++){
				sum+=this.w[i][j]*this.lI[j];
			}
			this.lO[i]=Activation.apply(sum,this.af);
		}
		return this.lO;
	}



	public double[] err(double[] dlt,double lr){
		double[] ndlt=new double[this.i];
		for (int i=0;i<this.i;i++){
			int sum=0;
			for (int j=0;j<this.n;j++){
				sum+=dlt[j]*this.w[j][i]*Activation.applyD(this.lI[i],this.af);
			}
			ndlt[i]=sum;
		}
		for (int i=0;i<this.n;i++){
			for (int j=0;j<this.i+1;j++){
				this.w[i][j]-=dlt[i]*this.lI[j]*lr;
			}
		}
		return ndlt;
	}



	private void _gen_w(){
		this.lI=new double[this.i+1];
		this.lI[this.i]=-1;
		this.lO=new double[this.n];
		this.w=w=new double[this.n][this.i+1];
		for (int i=0;i<this.n;i++){
			for (int j=0;j<this.i+1;j++) {
				this.w[i][j]=Math.random();
			}
		}
	}
}