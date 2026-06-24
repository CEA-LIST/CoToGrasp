import io
import os
from tkinter.messagebox import NO
import numpy as np
import torch
import torch
import trimesh as tm
from plotly import graph_objects as go
from PIL import Image
import colorsys

from utils.get_models import get_handmodel
from utils.rot6d import rot6d_to_matrix, euler_to_matrix

colors = [
    'blue', 'red', 'yellow', 'pink', 'gray', 'orange'
]

def plot_mesh(mesh, color='lightblue', opacity=1.0, scale=1.0):
    return go.Mesh3d(
        x=mesh.vertices[:, 0] * scale,
        y=mesh.vertices[:, 1] * scale,
        z=mesh.vertices[:, 2] * scale,
        i=mesh.faces[:, 0],
        j=mesh.faces[:, 1],
        k=mesh.faces[:, 2],
        color=color, opacity=opacity)

def plot_hand(verts, faces, color='lightpink', opacity=1.0):
    return go.Mesh3d(
        x=verts[:, 0],
        y=verts[:, 1],
        z=verts[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        color=color, opacity=opacity)

def plot_contact_points(pts, grad, color='lightpink'):
    pts = pts.detach().cpu().numpy()
    grad = grad.detach().cpu().numpy()
    grad = grad / np.linalg.norm(grad, axis=-1, keepdims=True)
    return go.Cone(x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], u=-grad[:, 0], v=-grad[:, 1], w=-grad[:, 2], anchor='tip',
                   colorscale=[(0, color), (1, color)], sizemode='absolute', sizeref=0.2, opacity=0.5)

def plot_frame(pose, pose_type='6d', size=0.05, name="frame", show=True):
    # Extract xyz and pose vectors
    if pose_type=='6d':
        xyz_pos = pose[:3]
        rot_mat = rot6d_to_matrix(torch.from_numpy(pose[3:9]))
    elif pose_type=='eul':
        xyz_pos = pose[:3]
        rot_mat = euler_to_matrix(torch.from_numpy(pose[3:6]))
    elif pose_type=='mat':
        xyz_pos = pose[:3, 3]
        rot_mat = torch.from_numpy(pose[:3, :3])

    # Define the local axes vectors and their colors
    axes_vectors_local = np.array([
        [size, 0, 0],  # x-axis
        [0, size, 0],  # y-axis
        [0, 0, size]   # z-axis
    ])
    
    # Define colors and names for the traces
    colors = ['red', 'green', 'blue']
    axis_names = ['X-axis', 'Y-axis', 'Z-axis']
    
    # Rotate the local axes to the global frame
    axes_vectors_global = rot_mat.cpu().numpy() @ axes_vectors_local
    
    # Create a list to hold the traces
    traces = []
    
    # Add a trace for each axis
    for i, (vec, color, axis_name) in enumerate(zip(axes_vectors_global.T, colors, axis_names)):
        traces.append(go.Scatter3d(
            # Define the start and end points of the line segment
            x=[xyz_pos[0], xyz_pos[0] + vec[0] / (size*10)],
            y=[xyz_pos[1], xyz_pos[1] + vec[1] / (size*10)],
            z=[xyz_pos[2], xyz_pos[2] + vec[2] / (size*10)],
            mode='lines',
            line=dict(
                color=color, # Use the single color directly
                width=size   # Use 'size' to control the line thickness
            ),
            showlegend=True,
            visible=show,
            name=f"{name}_{axis_name}",
        ))
    return traces

def plot_point_cloud(pts, color='lightblue', mode='markers', symbol='circle', size=3.5, name="pointcloud", show=True, text=None):
    """Symbol: ['circle', 'circle-open', 'cross', 'diamond', 'diamond-open', 'square', 'square-open', 'x']"""
    return go.Scatter3d(
        x=pts[:, 0],
        y=pts[:, 1],
        z=pts[:, 2],
        mode=mode,
        marker=dict(
            color=color,
            size=size,
            symbol=symbol
        ),
        name=name,
        showlegend=True,
        visible=show,
        text=text,
    )



rgb_color_map_24 = torch.tensor([
    [0.0, 0.0, 0.0], # 0: black
    [0.5643, 0.1272, 0.1272],
    [0.7812, 0.2527, 0.9573],
    [0.2487, 0.2487, 0.9516],
    [0.4981, 0.9595, 0.3443],
    [0.7297, 0.2128, 0.4712],
    [0.0873, 0.8467, 0.8467],
    [0.0930, 0.5842, 0.7480],
    [0.1919, 0.8702, 0.5310],
    [0.0209, 0.5694, 0.1580],
    [0.6525, 0.6525, 0.1939],
    [0.6439, 0.3060, 0.6439],
    [0.5966, 0.1954, 0.4963],
    [0.5513, 0.7607, 0.3419],
    [0.4981, 0.9595, 0.3443],
    [0.7823, 0.2552, 0.0795],
    [0.6082, 0.3484, 0.0886],
    [0.8034, 0.6345, 0.1278],
    [0.3647, 0.1738, 0.9372],
    [0.1501, 0.3185, 0.8237],
    [0.4282, 0.2463, 0.6101],
    [0.5052, 0.0103, 0.1341],
    [0.4414, 0.5201, 0.2053],
    [0.1241, 0.7926, 0.1241],
    [0.2773, 0.6154, 0.9535]
])

rgb_color_map_56 = torch.tensor([
    [0.0, 0.0, 0.0], # 0: black
    [0.5643, 0.1272, 0.1272],
    [0.7812, 0.2527, 0.9573],
    [0.2487, 0.2487, 0.9516],
    [0.7297, 0.2128, 0.4712],
    [0.0873, 0.8467, 0.8467],
    [0.0930, 0.5842, 0.7480],
    [0.1919, 0.8702, 0.5310],
    [0.0209, 0.5694, 0.1580],
    [0.6525, 0.6525, 0.1939],
    [0.6439, 0.3060, 0.6439],
    [0.5966, 0.1954, 0.4963],
    [0.5513, 0.7607, 0.3419],
    [0.4981, 0.9595, 0.3443],
    [0.7823, 0.2552, 0.0795],
    [0.6082, 0.3484, 0.0886],
    [0.8034, 0.6345, 0.1278],
    [0.3647, 0.1738, 0.9372],
    [0.1501, 0.3185, 0.8237],
    [0.4282, 0.2463, 0.6101],
    [0.5052, 0.0103, 0.1341],
    [0.4414, 0.5201, 0.2053],
    [0.1241, 0.7926, 0.1241],
    [0.2773, 0.6154, 0.9535],
    [0.5643, 0.1272, 0.1272],
    [0.7812, 0.2527, 0.9573],
    [0.2487, 0.2487, 0.9516],
    [0.7297, 0.2128, 0.4712],
    [0.0873, 0.8467, 0.8467],
    [0.0930, 0.5842, 0.7480],
    [0.1919, 0.8702, 0.5310],
    [0.0209, 0.5694, 0.1580],
    [0.6525, 0.6525, 0.1939],
    [0.6439, 0.3060, 0.6439],
    [0.5966, 0.1954, 0.4963],
    [0.5513, 0.7607, 0.3419],
    [0.4981, 0.9595, 0.3443],
    [0.7823, 0.2552, 0.0795],
    [0.6082, 0.3484, 0.0886],
    [0.8034, 0.6345, 0.1278],
    [0.3647, 0.1738, 0.9372],
    [0.1501, 0.3185, 0.8237],
    [0.4282, 0.2463, 0.6101],
    [0.5052, 0.0103, 0.1341],
    [0.4414, 0.5201, 0.2053],
    [0.1241, 0.7926, 0.1241],
    [0.2773, 0.6154, 0.9535],
    [0.5513, 0.7607, 0.3419],
    [0.4981, 0.9595, 0.3443],
    [0.7823, 0.2552, 0.0795],
    [0.6082, 0.3484, 0.0886],
    [0.8034, 0.6345, 0.1278],
    [0.3647, 0.1738, 0.9372],
    [0.1501, 0.3185, 0.8237],
    [0.4282, 0.2463, 0.6101],
    [0.5052, 0.0103, 0.1341],
    [0.4414, 0.5201, 0.2053],
    [0.1241, 0.7926, 0.1241],
])



generate_color_map = lambda N: torch.tensor([colorsys.hsv_to_rgb(i / N, torch.rand(1).item() * 0.5 + 0.5, torch.rand(1).item() * 0.5 + 0.5) for i in torch.randperm(N)])

color_map = lambda my_list: generate_color_map(int(np.max(my_list)) + 1)[np.array(my_list, dtype=int)]
color_map_24 = lambda my_list: rgb_color_map_24[np.array(my_list, dtype=int)]
color_map_56 = lambda my_list: rgb_color_map_56[np.array(my_list, dtype=int)]

def plot_point_cloud_label(pts, label, mode='markers', size=3.5, colors=None, name="pointcloud", show=True, text=None):
    text_displayed = text if text is not None else [f"{i}: {p}" for i, p in enumerate(pts)]
    return go.Scatter3d(
        x=pts[:, 0],
        y=pts[:, 1],
        z=pts[:, 2],
        mode=mode,
        marker=dict(
            # color=colors if colors is not None else color_map(label + np.abs(np.min(label))),
            color=colors if colors is not None else color_map_24(label),
            size=size,
        ),
        name=name,
        showlegend=True,
        visible=show,
        text=text_displayed,
        hovertemplate="<b>%{text}</b>"
    )


occ_cmap = lambda levels, thres=0.: [f"rgb({int(255)},{int(255)},{int(255)})" if x > thres else
                           f"rgb({int(0)},{int(0)},{int(0)})" for x in levels.tolist()]

def plot_point_cloud_occ(pts, color_levels=None):
    return go.Scatter3d(
        x=pts[:, 0],
        y=pts[:, 1],
        z=pts[:, 2],
        mode='markers',
        marker={
            'color': occ_cmap(color_levels),
            'size': 3,
            'opacity': 1
        }
    )


contact_cmap = lambda levels, thres=0.: [f"rgb({int(255 * (1 - x))},{int(255 * (1 - x))},{int(255 * (1 - x))})" if x >= thres else
                                         f"rgb({int(0)},{int(0)},{int(0)})" for x in levels.tolist()]

import matplotlib as mpl
bwr_cmap = mpl.colormaps['bwr']
spring_cmap = mpl.colormaps['spring']
custom_cmap_spring = lambda values: [mpl.colors.to_rgb(spring_cmap(v)) for v in values.tolist()]
custom_cmap_bwr = lambda values: [mpl.colors.to_rgb(bwr_cmap(v)) for v in values.tolist()]

def plot_point_cloud_cmap(pts, color_levels=None, size=5, name="pointcloud", show=True, text=None):
    return go.Scatter3d(
        x=pts[:, 0],
        y=pts[:, 1],
        z=pts[:, 2],
        mode='markers',
        name=name,
        marker={
            'color': custom_cmap_spring(color_levels),
            'size': size,
            'opacity': 1
        },
        visible=show,
        text=text,
        hovertemplate="<b>%{text}</b>"
    )

def plot_point_cloud_cmap_bwr(pts, color_levels=None, size=5, name="pointcloud", show=True, text=None):
    return go.Scatter3d(
        x=pts[:, 0],
        y=pts[:, 1],
        z=pts[:, 2],
        mode='markers',
        name=name,
        marker={
            'color': custom_cmap_bwr(color_levels),
            'size': size,
            'opacity': 1
        },
        visible=show,
        text=text,
        hovertemplate="<b>%{text}</b>" if text is not None else None
    )


normal_color_map = lambda levels: [f"rgb({int(255 * x[0])},{int(255 * x[1])},{int(255 * x[2])})" for x in levels.tolist()]


def plot_normal_map(pts, normal, size=5, name="pointcloud with normals", show=True, text=None):
    return go.Scatter3d(
        x=pts[:, 0],
        y=pts[:, 1],
        z=pts[:, 2],
        mode='markers',
        marker={
            'color': normal_color_map(np.abs(normal)),
            'size': size,
            'opacity': 1
        },
        name=name,
        visible=show,
        text=text,
    )


def plot_colored_mesh(mesh, opacity=1.0):
    return go.Mesh3d(
        x=mesh.vertices[:, 0],
        y=mesh.vertices[:, 1],
        z=mesh.vertices[:, 2],
        i=mesh.faces[:, 0],
        j=mesh.faces[:, 1],
        k=mesh.faces[:, 2],
        facecolor=normal_color_map(np.abs(mesh.face_normals)),
        opacity=opacity)


def plot_grasps(directory, tag, uuids, physics_guide, handcodes, contact_idx, ret_plots=False, save_html=True, include_contacts=True):
    handcode = handcodes[:, -1]
    hand_vertices = physics_guide.get_vertices(handcodes)
    hand_faces = physics_guide.hand_model.faces
    
    object_models = physics_guide.object_models
        
    if include_contacts:
        contact_points = []
        for ind in range(contact_idx.shape[1]):
            contact_point_vertices = torch.gather(
                hand_vertices, 1,
                contact_idx[:, ind].unsqueeze(-1).tile((1, 1, 3))
            )
            contact_points.append(contact_point_vertices.detach().cpu().numpy())
            
    hand_vertices = hand_vertices.detach().cpu().numpy()
    
    plots = []

    for batch_idx in range(hand_vertices.shape[0]):
        to_plot = []

        to_plot.append(plot_hand(hand_vertices[batch_idx], hand_faces))

        for obj_ind, obj in enumerate(object_models):
            to_plot.append(obj.get_plot(batch_idx))
            if include_contacts:
                to_plot.append(plot_point_cloud(contact_points[obj_ind][batch_idx], color=colors[obj_ind]))
        
        fig = go.Figure(to_plot)
        
        if save_html:
            fig.write_html(os.path.join(f"{ directory }", f"fig-{ str(uuids[ batch_idx ]) }-{ batch_idx }-{ tag }.html"))
        if ret_plots:
            plots.append(torch.from_numpy(np.asarray(Image.open(io.BytesIO(fig.to_image(format="png", width=1280, height=720))))))
            
    if ret_plots:
        return plots
    

def plot_mesh_from_name(dataset_object_name, color='lightblue', opacity=1.):
    dataset_name = dataset_object_name.split('+')[0]
    object_name = dataset_object_name.split('+')[1]
    mesh_path = os.path.join('data', 'object', dataset_name, object_name, f'{object_name}.stl')
    object_mesh = tm.load(mesh_path)
    return plot_mesh(object_mesh, color=color, opacity=opacity)


def plot_data(vis_data, grid=False, plot_title='', cam=None):
    """
    'allegro_right' --> cam_dir = dict(x=2, y=0, z=0) \n
    'shadowhand' --> cam_dir = dict(x=0, y=-2, z=0) \n
    'barrett' --> cam_dir = dict(x=2, y=0, z=2) \n
    """
    if not grid:
        d = dict(showgrid=False, zeroline=False, showline=False, ticks='', showaxeslabels=False, showticklabels=False, showbackground=False, title='')
    else:
        d = dict()

    fig = go.Figure(data=vis_data)
    fig.update_layout(
        title=plot_title,
        autosize=True,
        scene=dict(
            aspectmode='data',
            camera=dict(
                eye=cam if cam is not None else dict(x=0, y=-2, z=0),
                center=dict(x=0, y=0, z=0)
            ),
            xaxis=d,
            yaxis=d,
            zaxis=d,
        ),
    )
    try:
        fig.show(renderer='chromium')
    except:
        fig.show()

def save_data(vis_data, save_path, grid=False, plot_title='', cam=None):
    
    if not grid:
        d = dict(showgrid=False, zeroline=False, showline=False, ticks='', showaxeslabels=False, showticklabels=False, showbackground=False, title='')
    else:
        d = dict()
    fig = go.Figure(data=vis_data)
    fig.update_layout(
        title=plot_title,
        autosize=True,
        scene=dict(
            aspectmode='data',
            camera=dict(
                eye=cam if cam is not None else dict(x=0, y=-2, z=0),
                center=dict(x=0, y=0, z=0)
            ),
            xaxis=d,
            yaxis=d,
            zaxis=d,
        ),
    )
    fig.write_html(save_path)






from tqdm import tqdm
import imageio

def visu_traj(q_traj, object_trace, target_trace, robot_name, robot_color_map, vid_name, save=False):

    hand_model = get_handmodel(robot_name, device=q_traj.device)
    q_can = hand_model.canonical_pose.clone()
    frames = []
    for i in tqdm(range(q_traj.shape[0]), desc="Creating frames", leave=False):
        if i % 2 != 0:
            continue
        q = q_traj[i].unsqueeze(0).to(q_traj.device)
        pose = q_can.clone()
        pose[:, 9:] = q
        plotly_data = hand_model.get_plotly_data(pose, color_map=robot_color_map, opacity=0.5)
        # plotly_data is a list of plotly traces
        frames.append(go.Frame(data=plotly_data, name=str(i)))
        if i==0:
            if target_trace is not None:
                plotly_data0 = [object_trace] + [target_trace] + plotly_data
            else:
                plotly_data0 = [object_trace] + plotly_data

    for frame in frames:
        if target_trace is not None:
            frame.data = (object_trace,) + (target_trace,) + frame.data
        else:
            frame.data = (object_trace,) + frame.data

    # Create the figure with frames and initial data
    fig = go.Figure(
        data=plotly_data0,
        frames=frames,
        layout=go.Layout(
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    y=0.85,  # Move buttons lower (was 1)
                    x=1.15,
                    xanchor="right",
                    yanchor="top",
                    pad=dict(t=30),  # Add margin above buttons
                    buttons=[
                        dict(label="Play",
                             method="animate",
                             args=[None, {"frame": {"duration": 100, "redraw": True},
                                          "fromcurrent": True, "transition": {"duration": 0}}]),
                        dict(label="Pause",
                             method="animate",
                             args=[[None], {"frame": {"duration": 0, "redraw": False},
                                            "mode": "immediate",
                                            "transition": {"duration": 0}}]),
                        dict(label="Reset", 
                             method="animate",
                             args=[[str(0)], {"frame": {"duration": 0, "redraw": True},
                                             "mode": "immediate",
                                             "transition": {"duration": 0}}])
                    ]
                )
            ],
            sliders=[{
                "steps": [
                    {
                        "args": [[str(i)], {"frame": {"duration": 0, "redraw": True},
                                            "mode": "immediate",
                                            "transition": {"duration": 0}}],
                        "label": str(i),
                        "method": "animate"
                    } for i in range(q_traj.shape[0])
                ],
                "transition": {"duration": 0},
                "x": 0.1,
                "y": 0,
                "currentvalue": {"prefix": "Step: "},
                "len": 0.8
            }]
        )
    )

    if robot_name == 'allegro':
        cam_dir = dict(x=3, y=0, z=0)
    elif robot_name == 'shadowhand':
        cam_dir = dict(x=0, y=-3, z=0)
    elif robot_name == 'barrett':
        cam_dir = dict(x=2, y=0, z=2)
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode='data',
            camera=dict(
                eye=cam_dir,
                center=dict(x=0, y=0, z=0)
            ),
        ),
        margin=dict(l=0, r=0, t=0, b=0)
    )

    print("Showing figure...")
    # try:
    #     fig.show(renderer="chromium")
    # except:
    fig.show()

    if save:
        # Create and save a video of the frames
        images = []
        for i, frame in tqdm(enumerate(frames), total=len(frames), desc="Saving frames to video", leave=False):
            if i % 2 != 0:
                continue
            fig_tmp = go.Figure(
                data=frame.data,
                layout=go.Layout(
                    scene=dict(
                        xaxis=dict(visible=False),
                        yaxis=dict(visible=False),
                        zaxis=dict(visible=False),
                        aspectmode='data',
                        camera=dict(
                            eye=cam_dir,
                            center=dict(x=0, y=0, z=0)
                        ),
                    ),
                )
            )
            fig_tmp.update_layout(title=f"Step {i}")
            # Use kaleido to export to image bytes
            img_bytes = fig_tmp.to_image(format="png", width=800, height=600, scale=2, engine="kaleido")
            # Directly read the image bytes into a numpy array using imageio
            img = imageio.v3.imread(img_bytes, extension=".png")
            images.append(img)
        
        # Save as video (mp4)
        video_path = f'optimization/{vid_name}.mp4'
        print("Saving video...", end='\r')
        imageio.mimsave(video_path, images, fps=5)

        print(f"Video saved to {video_path}")

