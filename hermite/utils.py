import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.gridspec as gridspec
plt.style.use('fivethirtyeight')
plt.rcParams['font.family']='Times New Roman'
plt.rcParams['font.size']=24

def hermite(p0,p1,u0,u1,num_points=100):
    t=np.linspace(0,1,num_points).reshape(-1,1)
    h00=2*t**3 - 3*t**2 + 1
    h10=t**3 - 2*t**2 + t
    h01=-2*t**3 + 3*t**2
    h11=t**3 - t**2
    return h00 * p0 + h10 * u0 + h01 * p1 + h11 * u1

def length(x):
    df=pd.DataFrame(x,columns=['x','y','z'])
    disc_dst=np.sqrt((df['x'].diff()**2) + (df['y'].diff()**2) + (df['z'].diff()**2))
    lngth=disc_dst.sum()
    return lngth

def plotter(x,p0,p1,u0,u1,cad=None,tgt=None,prt=None,pos=0.7):
    fig=plt.figure(figsize=(20,20))
    gs=gridspec.GridSpec(2,3,figure=fig,height_ratios=[1,2])

    # xy plane
    ax1=fig.add_subplot(gs[0,0],projection='3d')
    ax1.plot(x[:,0],x[:,1],x[:,2],label='Unoptimized hermite Spline')
    ax1.scatter([p0[0],p1[0]],[p0[1],p1[1]],[p0[2],p1[2]],c='red',s=50,label='Points')
    ax1.quiver(p0[0],p0[1],p0[2],u0[0],u0[1],u0[2],color='slategrey',label='Unit Vectors')
    ax1.quiver(p1[0],p1[1],p1[2],u1[0],u1[1],u1[2],color='slategrey')
    ax1.set_title('XY Plane')
    ax1.set_xlabel('X',labelpad=40)
    ax1.set_ylabel('Y',labelpad=40)
    ax1.set_zlabel('')
    ax1.set_zticks([])
    ax1.view_init(elev=90,azim=-90)

    # yz plane
    ax2=fig.add_subplot(gs[0,1],projection='3d')
    ax2.plot(x[:,0],x[:,1],x[:,2],label='Unoptimized hermite Spline')
    ax2.scatter([p0[0],p1[0]],[p0[1],p1[1]],[p0[2],p1[2]],c='red',s=50,label='Points')
    ax2.quiver(p0[0],p0[1],p0[2],u0[0],u0[1],u0[2],color='slategrey',label='Unit Vectors')
    ax2.quiver(p1[0],p1[1],p1[2],u1[0],u1[1],u1[2],color='slategrey')
    ax2.set_title('YZ Plane')
    ax2.set_xlabel('')
    ax2.set_ylabel('Y',labelpad=40)
    ax2.set_zlabel('Z',labelpad=40)
    ax2.set_xticks([])
    ax2.view_init(elev=0,azim=0)

    # xz plane
    ax3=fig.add_subplot(gs[0,2],projection='3d')
    ax3.plot(x[:,0],x[:,1],x[:,2],label='Unoptimized hermite Spline')
    ax3.scatter([p0[0],p1[0]],[p0[1],p1[1]],[p0[2],p1[2]],c='red',s=50,label='Points')
    ax3.quiver(p0[0],p0[1],p0[2],u0[0],u0[1],u0[2],color='slategrey',label='Unit Vectors')
    ax3.quiver(p1[0],p1[1],p1[2],u1[0],u1[1],u1[2],color='slategrey')
    ax3.set_title('XZ Plane')
    ax3.set_xlabel('X',labelpad=40)
    ax3.set_ylabel('')
    ax3.set_zlabel('Z',labelpad=40)
    ax3.set_yticks([])
    ax3.view_init(elev=0,azim=-90)
    
    # 3d view
    ax4=fig.add_subplot(gs[1,:],projection='3d')
    ax4.plot(x[:,0],x[:,1],x[:,2],label='Unoptimized hermite spline')
    ax4.scatter([p0[0],p1[0]],[p0[1],p1[1]],[p0[2],p1[2]],c='red',s=50,label='Points')
    ax4.quiver(p0[0],p0[1],p0[2],u0[0],u0[1],u0[2],color='slategrey',label='Unit vectors')
    ax4.quiver(p1[0],p1[1],p1[2],u1[0],u1[1],u1[2],color='slategrey')
    ax4.set_title('3D View')
    ax4.set_xlabel('X',labelpad=20)
    ax4.set_ylabel('Y',labelpad=20)
    ax4.set_zlabel('Z',labelpad=20)
    ax4.view_init(elev=45,azim=-45)

    if cad is not None:
        ax1.plot(cad[:,0],cad[:,1],cad[:,2])
        ax2.plot(cad[:,0],cad[:,1],cad[:,2])
        ax3.plot(cad[:,0],cad[:,1],cad[:,2])
        ax4.plot(cad[:,0],cad[:,1],cad[:,2],label='CAD')

    if tgt is not None:
        ax1.plot(tgt[:,0],tgt[:,1],tgt[:,2])
        ax2.plot(tgt[:,0],tgt[:,1],tgt[:,2])
        ax3.plot(tgt[:,0],tgt[:,1],tgt[:,2])
        ax4.plot(tgt[:,0],tgt[:,1],tgt[:,2],label='Arc length constrained hermite spline')

    if prt is not None:
        ax1.plot(prt[:,0],prt[:,1],prt[:,2])
        ax2.plot(prt[:,0],prt[:,1],prt[:,2])
        ax3.plot(prt[:,0],prt[:,1],prt[:,2])
        ax4.plot(prt[:,0],prt[:,1],prt[:,2],label='Prototype')

    ax4.legend(bbox_to_anchor=[pos,1])
    plt.tight_layout()
    plt.show()

def plane_normal(df, start, end):
    mp=df.iloc[len(df)//2].to_numpy()
    v1=end-start
    v2=mp-start
    normal=np.cross(v1, v2)
    return normal / np.linalg.norm(normal)
    
def rotate(crv1,crv2):
    p1_s=crv1.iloc[0].to_numpy()
    p1_e=crv1.iloc[-1].to_numpy()
    p2_s=crv2.iloc[0].to_numpy()
    p2_e=crv2.iloc[-1].to_numpy()
    trans=p1_s-p2_s
    crv2_trans=crv2+trans
    v1=p1_e-p1_s
    v2=p2_e-p2_s
    v1=v1/np.linalg.norm(v1)
    v2=v2/np.linalg.norm(v2)
    axs=np.cross(v2, v1)
    axs_norm=np.linalg.norm(axs)
    if axs_norm<1e-8:
        R=np.eye(3)
    else:
        axs=axs/axs_norm
        ang=np.arccos(np.clip(np.dot(v2, v1), -1.0, 1.0))
        K=np.array([
            [0, -axs[2], axs[1]],
            [axs[2], 0, -axs[0]],
            [-axs[1], axs[0], 0]
        ])
        R=np.eye(3)+np.sin(ang)*K+(1-np.cos(ang))*(K @ K)
    pts=crv2_trans.to_numpy()
    pts2=(R @ (pts-p1_s).T).T+p1_s
    crv2_algn=pd.DataFrame(pts2, columns=['X','Y','Z'])
    p2_s=crv2_algn.iloc[0].to_numpy()
    p2_e  =crv2_algn.iloc[-1].to_numpy()
    axs=p1_e-p1_s
    axs=axs/np.linalg.norm(axs)
    n1=plane_normal(crv1, p1_s, p1_e)
    n2=plane_normal(crv2_algn, p2_s, p2_e)
    dot=np.clip(np.dot(n2, n1), -1.0, 1.0)
    ang=np.arccos(dot)
    if np.isnan(ang) or abs(ang) < 1e-8:
        R=np.eye(3)
    else:
        K=np.array([
            [0, -axs[2], axs[1]],
            [axs[2], 0, -axs[0]],
            [-axs[1], axs[0], 0]
        ])
        R=np.eye(3)+np.sin(ang)*K+(1-np.cos(ang))*(K @ K)
    pts=crv2_algn.to_numpy()
    pts2=(R @ (pts-p1_s).T).T+p1_s
    crv2_rot=pd.DataFrame(pts2, columns=['x','y','z'])
    return crv2_rot