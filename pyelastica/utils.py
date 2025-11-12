import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.gridspec as gridspec
import matplotlib.animation as manimation
import plotly.graph_objects as go
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

def plotter(x,p0,p1,u0,u1,spl1=None,spl2=None,spl3=None,spl4=None,pos=0.7,cne=5,intr='s',x_label='Spline0',spl1_label='Spline1',spl2_label='Spline2',spl3_label='Spline3',spl4_label='Spline4'):
    if intr=='s':
        fig=plt.figure(figsize=(20,20))
        gs=gridspec.GridSpec(2,3,figure=fig,height_ratios=[1,2])

        # xy plane
        ax1=fig.add_subplot(gs[0,0],projection='3d')
        ax1.plot(x[:,0],x[:,1],x[:,2],label=x_label)
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
        ax2.plot(x[:,0],x[:,1],x[:,2],label=x_label)
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
        ax3.plot(x[:,0],x[:,1],x[:,2],label=x_label)
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
        ax4.plot(x[:,0],x[:,1],x[:,2],label=x_label)
        ax4.scatter([p0[0],p1[0]],[p0[1],p1[1]],[p0[2],p1[2]],c='red',s=50,label='Points')
        ax4.quiver(p0[0],p0[1],p0[2],u0[0],u0[1],u0[2],color='slategrey',label='Unit vectors')
        ax4.quiver(p1[0],p1[1],p1[2],u1[0],u1[1],u1[2],color='slategrey')
        ax4.set_title('3D View')
        ax4.set_xlabel('X',labelpad=20)
        ax4.set_ylabel('Y',labelpad=20)
        ax4.set_zlabel('Z',labelpad=20)
        ax4.view_init(elev=45,azim=-45)

        if spl1 is not None:
            ax1.plot(spl1[:,0],spl1[:,1],spl1[:,2])
            ax2.plot(spl1[:,0],spl1[:,1],spl1[:,2])
            ax3.plot(spl1[:,0],spl1[:,1],spl1[:,2])
            ax4.plot(spl1[:,0],spl1[:,1],spl1[:,2],label=spl1_label)

        if spl2 is not None:
            ax1.plot(spl2[:,0],spl2[:,1],spl2[:,2])
            ax2.plot(spl2[:,0],spl2[:,1],spl2[:,2])
            ax3.plot(spl2[:,0],spl2[:,1],spl2[:,2])
            ax4.plot(spl2[:,0],spl2[:,1],spl2[:,2],label=spl2_label)

        if spl3 is not None:
            ax1.plot(spl3[:,0],spl3[:,1],spl3[:,2])
            ax2.plot(spl3[:,0],spl3[:,1],spl3[:,2])
            ax3.plot(spl3[:,0],spl3[:,1],spl3[:,2])
            ax4.plot(spl3[:,0],spl3[:,1],spl3[:,2],label=spl3_label)
        
        if spl4 is not None:
            ax1.plot(spl4[:,0],spl4[:,1],spl4[:,2])
            ax2.plot(spl4[:,0],spl4[:,1],spl4[:,2])
            ax3.plot(spl4[:,0],spl4[:,1],spl4[:,2])
            ax4.plot(spl4[:,0],spl4[:,1],spl4[:,2],label=spl4_label)

        ax4.legend(bbox_to_anchor=[pos,1])
        plt.tight_layout()
        return fig,ax1,ax2,ax3,ax4
    
    elif intr=='d':
        fig = go.Figure()
        fig.add_trace(go.Scatter3d(x=x[:,0],y=x[:,1],z=x[:,2],mode='lines',name=x_label,line=dict(color='darkslategray',width=6)))
        fig.add_trace(go.Scatter3d(x=[p0[0]],y=[p0[1]],z=[p0[2]],mode='markers',name='P0',marker=dict(size=5,color='red')))
        fig.add_trace(go.Scatter3d(x=[p1[0]],y=[p1[1]],z=[p1[2]],mode='markers',name='P1',marker=dict(size=5,color='red')))
        fig.add_trace(go.Cone(x=[p0[0]],y=[p0[1]],z=[p0[2]],u=[u0[0]],v=[u0[1]],w=[u0[2]],colorscale='RdPu',sizemode='absolute',sizeref=cne,showscale=False,name='Unit vector at P0',showlegend=True))
        fig.add_trace(go.Cone(x=[p1[0]],y=[p1[1]],z=[p1[2]],u=[u1[0]],v=[u1[1]],w=[u1[2]],colorscale='RdPu',sizemode='absolute',sizeref=cne,showscale=False,name='Unit vector at P1',showlegend=True))

        if spl1 is not None:
            fig.add_trace(go.Scatter3d(x=spl1[:,0],y=spl1[:,1],z=spl1[:,2],mode='lines',name=spl1_label,line=dict(color='dodgerblue',width=6)))

        if spl2 is not None:
            fig.add_trace(go.Scatter3d(x=spl2[:,0],y=spl2[:,1],z=spl2[:,2],mode='lines',name=spl2_label,line=dict(color='darkorange',width=6)))

        if spl3 is not None:
            fig.add_trace(go.Scatter3d(x=spl3[:,0],y=spl3[:,1],z=spl3[:,2],mode='lines',name=spl3_label,line=dict(color='forestgreen',width=6))) 
        
        if spl4 is not None:
            fig.add_trace(go.Scatter3d(x=spl4[:,0],y=spl4[:,1],z=spl4[:,2],mode='lines',name=spl4_label,line=dict(color='hotpink',width=6))) 

        fig.update_layout(title='3D Projection',scene=dict(xaxis_title='X',yaxis_title='Y',zaxis_title='Z',camera=dict(eye=dict(x=1,y=-1,z=1))),font=dict(size=15,family='Times New Roman'),autosize=False,height=700,width=1200,legend=dict(x=0.02,y=0.98),margin=dict(l=20,r=20,t=40,b=20))
        fig.update_scenes(camera_projection_type='orthographic')
        return fig



def plane_normal(df, start, end):
    mp=df.iloc[len(df)//2].to_numpy()
    v1=end-start
    v2=mp-start
    normal=np.cross(v1, v2)
    return normal / np.linalg.norm(normal)
    
def rotate(crv1,crv2,x0,x1):
    # translation
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
    x0=R @ x0
    x1=R @ x1
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
    x0=R @ x0
    x1=R @ x1
    pts2=(R @ (pts-p1_s).T).T+p1_s
    crv2_rot=pd.DataFrame(pts2, columns=['x','y','z'])
    return crv2_rot,x0,x1

def vector_compare(p0,p1,u0,u1,v0,v1):
    print(f'Difference in angle between CAD & Prototype @P0 : {float(np.degrees(np.arccos(np.dot(u0,v0)))):.3f}')
    print(f'Difference in angle between CAD & Prototype @P1 : {float(np.degrees(np.arccos(np.dot(u1,v1)))):.3f}')
    fig = go.Figure()
    fig.add_trace(go.Cone(x=[p0[0]],y=[p0[1]],z=[p0[2]],u=[u0[0]],v=[u0[1]],w=[u0[2]],colorscale='RdPu',sizemode='absolute',sizeref=7,showscale=False,name='CAD @P0',showlegend=True))
    fig.add_trace(go.Cone(x=[p1[0]],y=[p1[1]],z=[p1[2]],u=[u1[0]],v=[u1[1]],w=[u1[2]],colorscale='RdPu',sizemode='absolute',sizeref=7,showscale=False,name='CAD @P1',showlegend=True))
    fig.add_trace(go.Cone(x=[p0[0]],y=[p0[1]],z=[p0[2]],u=[v0[0]],v=[v0[1]],w=[v0[2]],colorscale='Greys',sizemode='absolute',sizeref=7,showscale=False,name='Prototype @P0',showlegend=True))
    fig.add_trace(go.Cone(x=[p1[0]],y=[p1[1]],z=[p1[2]],u=[v1[0]],v=[v1[1]],w=[v1[2]],colorscale='Greys',sizemode='absolute',sizeref=7,showscale=False,name='Prototype @P1',showlegend=True))
    fig.update_layout(title='Comparing Vectors',scene=dict(xaxis_title='X',yaxis_title='Y',zaxis_title='Z',camera=dict(eye=dict(x=1,y=-1,z=1))),font=dict(size=15,family='Times New Roman'),autosize=False,height=700,width=800,legend=dict(x=0.02,y=0.98),margin=dict(l=20,r=20,t=40,b=20))
    fig.update_scenes(camera_projection_type='orthographic')
    fig.show()

def plot_video3D(
    plot_params: dict,xlims,ylims,zlims,video_name:str="video.mp4",fps:int=25,
) -> None:
    t=np.array(plot_params["time"])
    positions_over_time=np.array(plot_params["position"])

    base_position,base_orientation=[],[]
    for p,q in plot_params["base_pose"]:
        base_position.append(p)
        base_orientation.append(q)

    FFMpegWriter=manimation.writers["ffmpeg"]
    metadata=dict(title="Movie Test",artist="Matplotlib",comment="Movie support!")
    writer=FFMpegWriter(fps=fps,metadata=metadata)

    fig=plt.figure()
    ax=fig.add_subplot(111,projection="3d")
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.set_zlim(zlims)
    ax.view_init(elev=25,azim=-45)
    ax.set_aspect("equal")
    rod_lines_3d=ax.plot(
        *positions_over_time[0],
        linewidth=3,
    )[0]
    with writer.saving(fig,video_name,dpi=100):
        for time in range(1,len(t) - 1):
            rod_lines_3d.set_xdata(positions_over_time[time][0])
            rod_lines_3d.set_ydata(positions_over_time[time][1])
            rod_lines_3d.set_3d_properties(positions_over_time[time][2])
            writer.grab_frame()
    plt.close(fig)
    print("Video saved as",video_name)

def dir_unitvectr(v):
    d3=v/np.linalg.norm(v)
    v_temp=np.array([0.0,1.0,0.0])
    if np.abs(np.dot(d3,v_temp)) > 0.999:
        v_temp=np.array([1.0,0.0,0.0])
    d1=np.cross(v_temp,d3)
    d1/=np.linalg.norm(d1)
    d2=np.cross(d3,d1)
    return np.column_stack([d1,d2,d3])