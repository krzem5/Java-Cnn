package com.krzem.cnn;



import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.List;



public class CNN{
	public int inW;
	public int inH;
	public List<Layer> layers;
	public List<FullyConnectedLayer> fclayers;
	public int fcW;
	public int fcD;
	public String[] cL;
	public String af;
	public double lr;



	public CNN(int inW,int inH,List<Layer> layers,int fcW,int fcD,String[] cL,String af,double lr){
		this.inW=inW;
		this.inH=inH;
		this.layers=layers;
		this.fcW=fcW;
		this.fcD=fcD;
		this.af=af;
		this.cL=cL;
		this.lr=lr;
		this._gen_fc();
	}



	public String classify(BufferedImage img){
		int[][][] dt=Dataset.convert(img);
		double[] o=this._predict(dt);
		double max=-1;
		int bestidx=-1;
		for (int i=0;i<o.length;i++){
			if (o[i]>max){
				max=o[i];
				bestidx=i+0;
			}
		}
		return this.cL[bestidx];
	}



	public void train(Dataset dt,Dataset test,int minE,int maxE,boolean log){
		dt.shuffle();
		double ca=0;
		double pa=0;
		if (log==true){
			System.out.println(String.format("Start accuracy: %f",this.test(test)));
		}
		for (int e=0;e<maxE;e++){
			this._train(dt,log);
			ca=this.test(test);
			if (log==true){
				double ta=this.test(dt);
				int w=40+String.format("%d%d%f%f",e+1,maxE,ta,ca).length();
				String b="";
				for (int i=0;i<w;i++){
					b+="=";
				}
				System.out.println(String.format("%s\nEpoch %d/%d complete (train_acc=%f, test_acc=%f)\n%s",b,e+1,maxE,ta,ca,b));
			}
			if (ca<pa&&e>=minE+1){
				break;
			}
			pa=ca+0;
		}
	}



	public double test(Dataset dt){
		int acc=0;
		for (int i=0;i<dt.size();i++){
			double[] o=this._predict(dt.getI(i));
			double m=-Double.MAX_VALUE;
			int mi=-1;
			for (int j=0;j<o.length;j++){
				if (o[j]>m){
					m=o[j];
					mi=j+0;
				}
			}
			if (this.cL[mi].equals(dt.getL(i))){
				acc++;
			}
		}
		return (double)(acc)/dt.size()*100;
	}



	private void _gen_fc(){
		int ow=this.inW+0;
		int oh=this.inH+0;
		int o=3;
		for (Layer l:this.layers){
			o=l.calc_o(o);
			ow=l.calc_ow(ow);
			oh=l.calc_oh(oh);
		}
		this.fclayers=new ArrayList<FullyConnectedLayer>();
		int i=ow*oh*o;
		i*=(this.layers.size()>0&&this.layers.get(this.layers.size()-1).getClass().getName().equals("com.krzem.cnn.ConvolutionLayer")?((ConvolutionLayer)this.layers.get(this.layers.size()-1)).convn():1);
		this.fclayers.add(new FullyConnectedLayer(i,this.fcW,this.af));
		for (int idx=0;idx<this.fcD-1;idx++){
			this.fclayers.add(new FullyConnectedLayer(this.fcW,this.fcW,this.af));
		}
		this.fclayers.add(new FullyConnectedLayer(this.fcW,this.cL.length,"sigmoid"));
	}



	private double[] _predict(int[][][] img){
		int[][][] dimg=(img[0].length/this.inH>1?Dataset.downsample(img,img[0].length/this.inH):img);
		double[][][] pl=this._img_to_plates(dimg);
		for (Layer l:this.layers){
			pl=l.out(pl);
		}
		double[] fcv=this._plates_to_vector(pl);
		for (FullyConnectedLayer fc:this.fclayers){
			fcv=fc.out(fcv);
		}
		return fcv;
	}



	private void _train(Dataset dt,boolean log){
		int l=-1;
		for (int i=0;i<dt.size();i++){
			if (log==true&&Math.floor((double)(i)/dt.size()*100)>l){
				l=(int)Math.floor((double)(i)/dt.size()*100);
				// System.out.println(String.format("%d%% complete...",l));
			}
			double[] o=this._predict(dt.getI(i));
			double[] to=this._label_to_vector(dt.getL(i));
			double[] fce=new double[o.length];
			for (int j=0;j<o.length;j++){
				fce[j]=(o[j]-to[j]);//*Activation.applyD(o[j],"sigmoid");
			}
			for (int j=this.fclayers.size()-1;j>=0;j--){
				fce=this.fclayers.get(j).err(fce,this.lr);
			}
			if (this.layers.size()>0){
				ConvolutionLayer lp=(ConvolutionLayer)this.layers.get(this.layers.size()-1);
				double[][][] ple=this._vector_to_plates(fce,lp.convh(),lp.convw());
				for (int j=this.layers.size()-1;j>=0;j--){
					ple=this.layers.get(j).err(ple,this.lr);
				}
			}
		}
	}



	private double[][][] _img_to_plates(int[][][] img){
		double[][][] o=new double[img.length][img[0].length][img[0][0].length];
		for (int i=0;i<img.length;i++){
			for (int j=0;j<img[i].length;j++){
				for (int k=0;k<img[i][0].length;k++){
					o[i][j][k]=(img[i][j][k])/255;
				}
			}
		}
		return o;
	}



	private double[] _plates_to_vector(double[][][] pl){
		double[] o=new double[pl.length*pl[0].length*pl[0][0].length];
		int off=0;
		for (int i=0;i<pl.length;i++){
			for (int j=0;j<pl[0].length;j++){
				System.arraycopy(pl[i][j],0,o,off,pl[i][j].length);
			}
		}
		return o;
	}



	private double[][][] _vector_to_plates(double[] v,int w,int h){
		double[][][] o=new double[v.length/(w*h)][h][w];
		for (int i=0;i<v.length/(w*h);i++){
			for (int j=0;j<h;j++){
				for (int k=0;k<w;k++){
					o[i][j][k]=v[i*(w*h)+j*h+k];
				}
			}
		}
		return o;
	}



	private double[] _label_to_vector(String l){
		double[] o=new double[this.cL.length];
		for (int i=0;i<this.cL.length;i++){
			if (this.cL[i].equals(l)){
				o[i]=1d;
				break;
			}
		}
		return o;
	}
}