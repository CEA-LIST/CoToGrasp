from isaac_validation.isaac_validator import IsaacValidator # Import isaacgym modules before importing torch to avoid segmentation fault
import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--robot_name', default='shadowhand', type=str, help='Name of the gripper to use')
    parser.add_argument('--chunk_size', default=10000, type=int, help='Number of grasps to validate at once')
    parser.add_argument('--object_path', default='', type=str, help='Path to the object directory')
    parser.add_argument('--object_urdf_path', default='', type=str, help='Path to the object URDF file')
    parser.add_argument('--verbose', default=0, type=int, help='Enable verbose output')
    args_ = parser.parse_args()
    return args_


if __name__ == "__main__":
    
    import torch

    args = get_parser()

    device = 'cuda'
    chunk_size = int(args.chunk_size)
    verbose = bool(args.verbose)

    simulator = IsaacValidator(
        robot_name=args.robot_name,
        batch_size=chunk_size,
        use_gui=False,
        use_controller=False,
        use_stiffness=True,
        debug_interval=0.01,
    )
    simulator.set_asset(
        object_path=args.object_path,
        object_file=args.object_urdf_path,
        scale=1.0,
    )
    simulator.create_envs()

    q_batch = torch.load('temp/isaac_q_batch.pt', map_location=device, weights_only=True)

    success, q_isaac = torch.zeros(q_batch.shape[0], dtype=torch.bool), torch.zeros_like(q_batch)
    
    for k in range(0, q_batch.shape[0], chunk_size):
        q_batch_chunk = q_batch[k:k+chunk_size].clone().detach()
        current_batch_size = q_batch_chunk.shape[0]
    
        # Pad the last chunk
        if current_batch_size < chunk_size:
            pad_size = chunk_size - current_batch_size
            padding = q_batch_chunk[-1].unsqueeze(0).repeat(pad_size, 1)
            q_input = torch.cat([q_batch_chunk, padding], dim=0)
        else:
            q_input = q_batch_chunk

        simulator.set_actor_pose_dof(q_input)
        success_chunk, q_isaac_chunk = simulator.run_sim(verbose=verbose, desc=f"[Isaac/Run] {k+current_batch_size}/{q_batch.shape[0]}")
        success[k:k+current_batch_size] = success_chunk[:current_batch_size]
        q_isaac[k:k+current_batch_size] = q_isaac_chunk[:current_batch_size]
        
    simulator.destroy()
    
    # Store results => q_isaac, success
    torch.save((success, q_isaac), 'temp/isaac_results.pt')
