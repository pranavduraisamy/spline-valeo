def gravity_looper(psit=1,case=1,base_radius=2.43,density=1.23e-6,youngs_modulus=0.003605,twist=0,intr='d',pfix='',csv=1,html=0):
    import numpy as np
    import pandas as pd
    import utils
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    import plotly.io as pio
    import os
    from typing import Callable,Any
    from scipy.spatial.transform import Rotation,Slerp
    from elastica.modules import BaseSystemCollection,Constraints,Forcing,Damping,CallBacks
    from elastica.rod.cosserat_rod import CosseratRod
    from elastica.boundary_conditions import OneEndFixedBC,ConstraintBase
    from elastica.external_forces import GravityForces
    from elastica.dissipation import AnalyticalLinearDamper
    from elastica.callback_functions import CallBackBaseClass,defaultdict
    from elastica.timestepper.symplectic_steppers import PositionVerlet
    from elastica.timestepper import integrate

    # Simulator
    class Simulator(
        BaseSystemCollection,Constraints,Forcing,Damping,CallBacks
    ):
        pass

    # Data input
    df=pd.read_csv('../data/2810/p'+str(psit)+'-case'+str(case)+'.csv')
    pts=df.iloc[10:,2:].reset_index(drop=True)
    length=utils.length(pts.values)
    simulator=Simulator()
    n_elements=100 
    base_length=length 
    base_radius=base_radius 
    start_pos=df.iloc[5,2:].values.astype(float)
    direction=np.array([0.0,1.0,0.0]) 
    normal=np.array([0.0,0.0,1.0]) 
    density=density 
    youngs_modulus=youngs_modulus
    shear_modulus=youngs_modulus/(2.0*(1.0+0.49))

    # Rod def
    tube=CosseratRod.straight_rod(
        n_elements=n_elements,
        start=start_pos,
        direction=direction,
        normal=normal,
        base_length=base_length,
        base_radius=base_radius,
        density=density,
        youngs_modulus=youngs_modulus,
        shear_modulus=shear_modulus
    )

    # BCs
    # grav
    simulator.append(tube)
    simulator.add_forcing_to(tube).using(
        GravityForces,
        acc_gravity=np.array([0.0,0.0,-9.81e-3])
    )

    # damp
    dmp_const=0.3
    dt=1e-3
    simulator.dampen(tube).using(
        AnalyticalLinearDamper,
        damping_constant=dmp_const,
        time_step=dt
    )
    print('Added dampner to the simulation')

    # Fixed BC
    simulator.constrain(tube).using(
        OneEndFixedBC,
        constrained_position_idx=(0,),
        constrained_director_idx=(0,)
    )
    print('Added FixedBC')

    # movin end
    class EndpointKinematicConstraint(ConstraintBase):
        def __init__(self,
                    node_idx: int,
                    elem_idx: int,
                    target_position_function: Callable[[float],np.ndarray],
                    target_director_function: Callable[[float],np.ndarray],
                    target_velocity_function: Callable[[float],np.ndarray],
                    target_omega_function: Callable[[float],np.ndarray],
                    **kwargs):
            super().__init__(**kwargs)
            self.node_idx=node_idx
            self.elem_idx=elem_idx        
            self.pos_func=target_position_function
            self.dir_func=target_director_function
            self.vel_func=target_velocity_function
            self.omg_func=target_omega_function

        def constrain_values(self,system: Any,time: float) -> None:
            target_pos=self.pos_func(time)
            target_dir=self.dir_func(time)
            system.position_collection[...,self.node_idx]=target_pos
            system.director_collection[...,self.elem_idx]=target_dir
            
        def constrain_rates(self,system: Any,time: float) -> None:
            target_vel=self.vel_func(time)
            target_omg=self.omg_func(time)
            system.velocity_collection[...,self.node_idx]=target_vel
            system.omega_collection[...,self.elem_idx]=target_omg

    class TrajectoryRamp:
        def __init__(self,
                    start_pos: np.ndarray,
                    end_pos: np.ndarray,
                    start_dir: np.ndarray,
                    end_dir: np.ndarray,
                    ramp_time: float):
            
            self.start_pos=start_pos
            self.end_pos=end_pos
            self.pos_diff=end_pos - start_pos        
            self.ramp_time=ramp_time
            self.const_vel=self.pos_diff / ramp_time
            key_rots=Rotation.from_matrix([start_dir,end_dir])
            key_times=[0,ramp_time]
            self.slerp=Slerp(key_times,key_rots)
            self.const_omg=np.zeros((3,))
            
        def _get_ramp_factor(self,time: float) -> float:
            if time < 0:
                return 0.0
            elif time > self.ramp_time:
                return 1.0
            else:
                return time / self.ramp_time

        def get_position(self,time: float) -> np.ndarray:
            factor=self._get_ramp_factor(time)
            return self.start_pos + self.pos_diff * factor

        def get_director(self,time: float) -> np.ndarray:
            clamped_time=max(0,min(time,self.ramp_time))
            return self.slerp(clamped_time).as_matrix()

        def get_velocity(self,time: float) -> np.ndarray:
            if 0 < time <= self.ramp_time:
                return self.const_vel
            else:
                return np.zeros((3,))

        def get_omega(self,time: float) -> np.ndarray:
            if 0 < time <= self.ramp_time:
                return self.const_omg
            else:
                return np.zeros((3,))
            
    class TrajectoryCircle:
        def __init__(self,center: np.ndarray,radius: float,freq: float):
            self.center=center
            self.radius=radius
            self.omega_val=2.0 * np.pi * freq

        def get_position(self,time: float) -> np.ndarray:
            x=self.center + self.radius * np.cos(self.omega_val * time)
            y=self.center + self.radius * np.sin(self.omega_val * time)
            return np.array([x,y,self.center])
        
        def get_velocity(self,time: float) -> np.ndarray:
            vx=-self.radius * self.omega_val * np.sin(self.omega_val * time)
            vy=self.radius * self.omega_val * np.cos(self.omega_val * time)
            return np.array([vx,vy,0.0])

        def get_director(self,time: float) -> np.ndarray:
            return np.identity(3)

        def get_omega(self,time: float) -> np.ndarray:
            return np.zeros((3,))
        
    # Define targets
    init_pos=tube.position_collection[...,-1].copy()
    init_dir=tube.director_collection[...,-1].copy()
    tgt_pos=df.iloc[6,2:].values
    tgt_dir=utils.dir_unitvectr(df.iloc[9,2:].values).T
    theta=np.deg2rad(twist)
    tgt_dir[0]=(tgt_dir[0]*np.cos(theta)+np.cross(tgt_dir[2],tgt_dir[0])*np.sin(theta)+tgt_dir[2]*np.dot(tgt_dir[2],tgt_dir[0])*(1-np.cos(theta)))
    tgt_dir[1]=(tgt_dir[1]*np.cos(theta)+np.cross(tgt_dir[2],tgt_dir[1])*np.sin(theta)+tgt_dir[2]*np.dot(tgt_dir[2],tgt_dir[1])*(1-np.cos(theta)))
    ramp_time=350

    trajectory=TrajectoryRamp(
        init_pos,tgt_pos,init_dir,tgt_dir,ramp_time
    )

    simulator.constrain(tube).using(
        EndpointKinematicConstraint,
        node_idx=-1,
        elem_idx=-1,
        target_position_function=trajectory.get_position,
        target_director_function=trajectory.get_director,
        target_velocity_function=trajectory.get_velocity,
        target_omega_function=trajectory.get_omega
    )

    # cb
    class TubeCallback(CallBackBaseClass):
        def __init__(self,step_skip: int,callback_params: dict):
            CallBackBaseClass.__init__(self)
            self.every=step_skip
            self.callback_params=callback_params

        def make_callback(self,system: Any,time: int,current_step: int):
            if current_step % self.every == 0:
                self.callback_params["time"].append(time)
                self.callback_params["step"].append(current_step)
                self.callback_params["position"].append(system.position_collection.copy())
                self.callback_params["directors"].append(system.director_collection.copy())
                self.callback_params["length"].append(system.rest_lengths.copy())
                self.callback_params["radius"].append(system.radius.copy())
                self.callback_params["velocity"].append(system.velocity_collection.copy())
                self.callback_params["avg_velocity"].append(system.compute_velocity_center_of_mass())
                self.callback_params["com"].append(system.compute_position_center_of_mass())
                self.callback_params["curvature"].append(system.kappa.copy())
                return
    cb_data=defaultdict(list)
    dt_intrvl=100
    simulator.collect_diagnostics(tube).using(
        TubeCallback,
        step_skip=dt_intrvl,
        callback_params=cb_data
    )

    # finalize
    simulator.finalize()
    print('Simulator finalized')

    # solver
    timestepper=PositionVerlet()
    final_time=750 
    total_steps=int(final_time/dt)

    print("Running simulation...")
    integrate(timestepper,simulator,final_time,total_steps)
    print("Simulation finished.")

    # plotters
    position_history=np.array(cb_data["position"])
    final_rod_position=position_history[-1]

    df=pd.read_csv('../data/2810/p'+str(psit)+'-case'+str(case)+'.csv')
    prt=df.iloc[10:,2:].reset_index(drop=True)
    p0=df.iloc[5,2:].values.astype(float)   
    p1=df.iloc[6,2:].values.astype(float)   
    u0=df.iloc[8,2:].values.astype(float)*df.iloc[2,1]   
    u1=df.iloc[9,2:].values.astype(float)*df.iloc[3,1]   
    siml=pd.DataFrame(final_rod_position.T,columns=['X','Y','Z'])

    if intr=='d':
        fig=utils.plotter(prt.values,p0,p1,u0,u1,x_label='Prototype',spl1=siml.values,spl1_label='Simulated',cne=6,intr='d')
        a=df.iloc[59,2:]
        b=siml.iloc[50]
        fig.add_trace(go.Scatter3d(x=[a[0],b[0]],y=[a[1],b[1]],z=[a[2],b[2]],mode='lines',name='Error',line=dict(color='red',width=6,dash='longdash')))
        fig.show()
        if html==1:
            if not os.path.exists(pfix+'html/'):
                os.makedirs(pfix+'html/')
            pio.write_html(fig,pfix+'html/p'+str(psit)+'-case'+str(case)+'.html')

    if intr=='s':
        fig,ax1,ax2,ax3,ax4=utils.plotter(prt.values,p0,p1,u0,u1,x_label='Prototype',spl1=siml.values,spl1_label='Simulated',cne=6,intr='s')
        ax4.plot([a[0],b[0]],[a[1],b[1]],[a[2],b[2]],label='Error',linestyle='--')
        ax4.legend()
        fig.show()

    if csv==1:
        if not os.path.exists(pfix+'data/'):
            os.makedirs(pfix+'data/')
        siml.to_csv(pfix+'data/p'+str(psit)+'-case'+str(case)+'.csv',index=False)
    
    print('Task completed')