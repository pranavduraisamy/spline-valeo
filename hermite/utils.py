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

def plotter(x,p0,p1,u0,u1,padd=2):
    fig=plt.figure(figsize=(20,20))
    gs=gridspec.GridSpec(2,3,figure=fig,height_ratios=[1,2])
    lmt0=float(min(np.concat((p0,p1))))-padd
    lmt1=float(max(np.concat((p0,p1))))+padd

    # xy plane
    ax1=fig.add_subplot(gs[0,0])
    ax1.plot(x[:,0],x[:,1],label='Hermite Spline')
    ax1.scatter([p0[0],p1[0]],[p0[1],p1[1]],c='red',label='Points')
    ax1.quiver([p0[0],p1[0]],[p0[1],p1[1]],[u0[0],u1[0]],[u0[1],u1[1]],color='slategrey',angles='xy',scale_units='xy',scale=1,label='Unit Vectors')
    ax1.set_title('XY Plane')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_aspect('equal',adjustable='box')
    ax1.set_xlim(lmt0,lmt1)
    ax1.set_ylim(lmt0,lmt1)
    ax1.grid(True)
    ax1.legend()

    # yz plane
    ax2=fig.add_subplot(gs[0,1])
    ax2.plot(x[:,1],x[:,2],label='Spline')
    ax2.scatter([p0[1],p1[1]],[p0[2],p1[2]],c='red')
    ax2.quiver([p0[1],p1[1]],[p0[2],p1[2]],[u0[1],u1[1]],[u0[2],u1[2]],color='slategrey',angles='xy',scale_units='xy',scale=1)
    ax2.set_title('YZ Plane')
    ax2.set_xlabel('Y')
    ax2.set_ylabel('Z')
    ax2.set_aspect('equal',adjustable='box')
    ax2.set_xlim(lmt0,lmt1)
    ax2.set_ylim(lmt0,lmt1)
    ax2.grid(True)

    # zx plane
    ax3=fig.add_subplot(gs[0,2])
    ax3.plot(x[:,2],x[:,0],label='Spline')
    ax3.scatter([p0[2],p1[2]],[p0[0],p1[0]],c='red')
    ax3.quiver([p0[2],p1[2]],[p0[0],p1[0]],[u0[2],u1[2]],[u0[0],u1[0]],color='slategrey',angles='xy',scale_units='xy',scale=1)
    ax3.set_title('ZX Plane')
    ax3.set_xlabel('Z')
    ax3.set_ylabel('X')
    ax3.set_aspect('equal',adjustable='box')
    ax3.set_xlim(lmt0,lmt1)
    ax3.set_ylim(lmt0,lmt1)
    ax3.grid(True)
    
    # 3d view
    ax4=fig.add_subplot(gs[1,:],projection='3d')
    ax4.plot(x[:,0],x[:,1],x[:,2],label='Hermite Spline')
    ax4.scatter([p0[0],p1[0]],[p0[1],p1[1]],[p0[2],p1[2]],c='red',s=50,label='Points')
    ax4.quiver(p0[0],p0[1],p0[2],u0[0],u0[1],u0[2],color='slategrey',label='Unit Vectors')
    ax4.quiver(p1[0],p1[1],p1[2],u1[0],u1[1],u1[2],color='slategrey')
    ax4.set_title('3D View')
    ax4.set_xlabel('X',labelpad=20)
    ax4.set_ylabel('Y',labelpad=20)
    ax4.set_zlabel('Z',labelpad=20)
    #ax4.legend()
    ax4.view_init(elev=45,azim=-45)
    plt.tight_layout()
    plt.show()
