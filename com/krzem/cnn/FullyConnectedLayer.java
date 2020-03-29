package com.krzem.cnn;



public class FullyConnectedLayer{
	public int i;
	public int n;
	public String af;
	private double[][] w;
	private double[] lI;



	public FullyConnectedLayer(int i,int n,String af){
		this.i=i;
		this.n=n;
		this.af=af;
		this.w=this._gen_w(i,n);
	}



	public double[] out(double[] in){
		// System.out.println("=======================");
		System.arraycopy(in,0,this.lI,0,in.length);
		// for (double k:in){
		// 	System.out.println(k);
		// }
		double[] o=new double[this.n];
		for (int i=0;i<this.n;i++){
			for (int j=0;j<this.lI.length;j++){
				o[i]+=this.w[i][j]*this.lI[j];
			}
			// System.out.print("A: ");
			// System.out.println(o[i]);
			o[i]=Activation.apply(o[i],this.af);
		}
		return o;
	}



	public double[] err(double[] dlt,double lr){
		double[] o=new double[this.i];
		for (int i=0;i<this.i;i++){
			for (int j=0;j<this.n;j++){
				o[i]+=dlt[j]*this.w[j][i]*Activation.applyD(this.lI[i],this.af);
				this.w[j][i]-=dlt[j]*this.lI[i]*lr;
				if (((Double)dlt[j]).isNaN()){
					System.out.print("CCC: ");
					System.out.println(j);
					for (double k:dlt){
						System.out.println(k);
					}
					System.out.println("CCC: ");
					System.exit(1);
				}
			}
		}
		return o;
	}



	private double[][] _gen_w(int in,int n){
		this.lI=new double[in+1];
		this.lI[in]=1;
		double[][] o=new double[n][in+1];
		for (int i=0;i<n;i++){
			for (int j=0;j<in+1;j++) {
				o[i][j]=Random.nextG();
			}
		}
		return o;
	}
}