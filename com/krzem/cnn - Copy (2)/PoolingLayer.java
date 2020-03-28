package com.krzem.cnn - Copy (2);



import java.lang.Math;
import java.util.ArrayList;
import java.util.List;



public class PoolingLayer implements Layer{
	public int w;
	public int h;	
	private List<boolean[][]> max;



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
		return iw/w+(iw%w>0?1:0);
	}



	@Override
	public int calc_oh(int ih){
		return ih/h+(ih%h>0?1:0);
	}



	@Override
	public List<ImagePlate> out(List<ImagePlate> in){
		if (this.max==null){
			this.max=new ArrayList<boolean[][]>();
			for (int i=0;i<in.size();i++){
				this.max.add(new boolean[in.get(i).h][in.get(i).w]);
			}
		}
		List<ImagePlate> o=new ArrayList<ImagePlate>();
		for (int i=0;i<in.size();i++){
			o.add(this._max_pool(in.get(i),this.max.get(i),this.w,this.h));
		}
		in=null;
		return o;
	}



	@Override
	public List<ImagePlate> err(List<ImagePlate> g,double lr){
		List<ImagePlate> o=new ArrayList<ImagePlate>();
		for (int i=0;i<g.size();i++){
			ImagePlate e=g.get(i);
			double[][] eo=new double[this.max.get(0).length][this.max.get(0)[0].length];
			boolean[][] max=this.max.get(i);
			for (int j=0;j<this.max.get(0).length;j++){
				for (int k=0;k<this.max.get(0)[0].length;k++){
					eo[j][k]=(max[j][k]==true?e.get(j/this.h,k/this.w):0);
				}
			}
			o.add(new ImagePlate(eo));
		}
		return o;
	}



	private ImagePlate _max_pool(ImagePlate p,boolean[][] max,int w,int h){
		int ow=p.w/w+(p.w%w>0?1:0);
		int oh=p.h/h+(p.h%h>0?1:0);
		double[][] o=new double[oh][ow];
		for (int i=0;i<oh;i++){
			for (int j=0;j<ow;j++){
				int sk=Math.min(i*h,p.h-1);
				int sl=Math.min(j*w,p.w-1);
				int ek=Math.min(sk+h,p.h);
				int el=Math.min(sl+w,p.w);
				double maxv=Double.MIN_VALUE;
				int maxi=-1;
				int maxj=-1;
				for (int k=sk;k<ek;k++){
					for (int l=sl;l<el;l++){
						double v=p.get(k,l);
						if (v>maxv){
							maxv=v;
							maxi=k;
							maxj=l;
						}
					}
				}
				max[maxi][maxj]=true;
				o[i][j]=maxv;
			}
		}
		ImagePlate pl=new ImagePlate(o);
		o=null;
		return pl;
	}
}