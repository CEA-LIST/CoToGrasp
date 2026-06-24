import torch
import subprocess
import os
import json
from tqdm import tqdm
from termcolor import cprint, colored
import argparse
from datetime import datetime, timedelta
from tabulate import tabulate
from utils.constants import ROOT_PATH, DATA_PATH, ROBOTS_GRASPS_LABELS, IDX_2_GRASP_TYPE
from utils.rot6d import q_rot6d_to_q_euler

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--robot_name', default='shadowhand', type=str, help='Name of the gripper to use')
    parser.add_argument('--dataset', default='multidex', type=str, help='Dataset name if specific object')
    parser.add_argument('--file_name', default='', type=str, help='Specific file name')
    args_ = parser.parse_args()
    return args_

if __name__ == "__main__":
    args = get_parser()
    device = 'cuda'
    verbose = 1

    script_path = os.path.join(ROOT_PATH, 'isaac_validation/isaac_main.py')

    # Find last date string
    current_date = datetime.now()
    log_path = None
    for _ in range(30):
        date_str = current_date.strftime('%m%d%Y')
        potential_log_path = os.path.join(ROOT_PATH, 'logs_inference_grasps', f'{date_str}')
        if os.path.exists(potential_log_path):
            log_path = potential_log_path
            break
        current_date -= timedelta(days=1)
    
    assert log_path is not None, "No valid log path found in the last 30 days."
    if log_path:
        print(f"Using log path: {log_path}")
    all_files = os.listdir(log_path)
    if args.file_name != '':
        assert args.file_name in all_files, f"Specified file name {args.file_name} not found in log path."
        file_name = args.file_name
    else:
        file_name = f'{args.dataset}_{args.robot_name}_predicted_q.pt'
    # metadata, time_table = torch.load(os.path.join(log_path, file_name), map_location='cpu', weights_only=False)
    metadata = torch.load(os.path.join(log_path, file_name), map_location='cpu', weights_only=False)

    # cprint(f"******** [{args.dataset.upper()}/{args.robot_name.upper()} - Time Table]", 'magenta', attrs=['bold'])
    # cprint(time_table, 'magenta')

    with open(os.path.join(DATA_PATH, 'pointclouds', f'split_{args.dataset}.json'), 'r') as f:
        split_data = json.load(f)
    objects_list = split_data['test_split']
    
    # Create temporary directory for inter-process communication
    os.makedirs('temp', exist_ok=True)

    cprint(f"******** [{args.dataset.upper()}/{args.robot_name.upper()} - Isaac Validation]", 'yellow', attrs=['bold'])
    validation_results = []
    for idx, data in enumerate(tqdm(metadata, desc=f"Progress:", unit='objects')):

        object_name = data['object_name'].replace("+", "/")

        if object_name not in objects_list:
            continue

        object_path = os.path.join(DATA_PATH, 'urdf/objects')

        if args.dataset == 'multidex':
            object_name_split = object_name.split('/')
            object_urdf_path = f'{args.dataset}/{object_name_split[0]}/{object_name_split[1]}/coacd_decomposed_object_one_link.urdf'
        elif args.dataset == 'real_exp':
            object_urdf_path = f'{args.dataset}/{object_name}/coacd_decomposed_object_one_link.urdf'
        else:
            object_urdf_path = f'{args.dataset}/{object_name}.urdf'

        grasp_type = data['grasp_types']
        q_batch = data['predicted_q']

        # Send q_batch to Isaac Validator script
        torch.save(q_batch, 'temp/isaac_q_batch.pt')
        
        # Run the Isaac Validator script as a subprocess
        subprocess_args = [
            'python',
            script_path,
            '--robot_name', args.robot_name,
            '--chunk_size', '5000',
            '--object_path', object_path,
            '--object_urdf_path', object_urdf_path,
            '--verbose', str(verbose)
        ]

        ret = subprocess.run(subprocess_args, check=False) if verbose else subprocess.run(subprocess_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE) 

        if os.path.exists('temp/isaac_results.pt'): # Successful execution
            # Retrieve q_isaac, success from the subprocess output
            success, q_isaac = torch.load('temp/isaac_results.pt', map_location=device, weights_only=True)

            success_per_type = {t: {"success": 0, "total": 0} for t in ROBOTS_GRASPS_LABELS[args.robot_name].keys()}
            for j in range(q_batch.shape[0]):
                t = IDX_2_GRASP_TYPE[args.robot_name][grasp_type[j].item()]
                if success[j]:
                    success_per_type[t]['success'] += 1
                success_per_type[t]['total'] += 1

            success_num = success.sum().item()
            success_rate = success_num / q_batch.shape[0] * 100

            cprint(f"--- {idx}/{len(metadata)}: {object_name} --- {success_num}/{q_batch.shape[0]} ({success_rate:.2f} %) ", 'yellow')

            validation_results.append({
                'object_name': object_name.replace("/", "+"),
                'grasp_types': grasp_type,
                'predicted_q': q_batch,
                'q_isaac': q_isaac,
                'success': success,
                'success_per_type': success_per_type
            })

            os.remove('temp/isaac_results.pt')  # Clean up the results file

            if idx % 10 == 0:
                result_path = os.path.join(ROOT_PATH, 'logs_isaac', f'{date_str}')
                if not os.path.exists(result_path):
                    os.makedirs(result_path, exist_ok=False)
                file_name = f'{args.robot_name}_validation_results_{args.dataset}.pt'
                torch.save(validation_results, os.path.join(result_path, file_name))
        else:
            print(f"[Error] Isaac Validator failed for object {object_name}. Return code: {ret.returncode}")
            if verbose:
                # print(f"Stdout: {ret.stdout.decode('utf-8')}")
                # print(f"Stderr: {ret.stderr.decode('utf-8')}")
                stdout_text = ret.stdout.decode('utf-8') if ret.stdout else "No output"
                std_err_text = ret.stderr.decode('utf-8') if ret.stderr else "No error message"
                print(f"Stdout: {stdout_text}")
                print(f"Stderr: {std_err_text}")
                

    # Delete temporary files
    if os.path.exists('temp'):
        for temp_file in os.listdir('temp'):
            os.remove(os.path.join('temp', temp_file))
        os.rmdir('temp')

    # Save final results
    result_path = os.path.join(ROOT_PATH, 'logs_isaac', f'{date_str}')
    if not os.path.exists(result_path):
        os.makedirs(result_path, exist_ok=False)
    file_name = f'{args.robot_name}_validation_results_{args.dataset}.pt'
    torch.save(validation_results, os.path.join(result_path, file_name))
    print(f"[Info] Saved validation results to {os.path.join(result_path, file_name)}")

    # Prepare output strings
    output_lines = [f"\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"]
    output_lines.append(f"******** [{args.robot_name.upper()} - Results]")
    
    cprint(f"******** [{args.dataset.upper()}/{args.robot_name.upper()} - Results]", 'light_green', attrs=['bold'])    
    type_total_result = {t: {"success": 0, "total": 0} for t in ROBOTS_GRASPS_LABELS[args.robot_name].keys()}
    table_data, table_data_raw = [], []
    total_success, total_trials = 0, 0
    all_success_q = []
    for data in validation_results:
        object_name = data['object_name']
        success = int(data['success'].sum().item())
        trials = int(data['success'].shape[0])
        type_details = data['success_per_type']
        
        all_success_q.append(data['q_isaac'][data['success']])

        row_raw = []
        type_row = []
        for g_type in ROBOTS_GRASPS_LABELS[args.robot_name].keys():
            if g_type not in type_details:
                type_row.append("-")
                row_raw.append("-")
                continue
            g_success = type_details[g_type]['success']
            g_total = type_details[g_type]['total']
            
            if g_success > 0 and g_total > 0:
                success += g_success
                trials += g_total

                type_total_result[g_type]['success'] += g_success
                type_total_result[g_type]['total'] += g_total
                
                g_rate = 100.0 * g_success / g_total if g_total > 0 else 0.0
                g_color = 'light_red' if g_rate < 50 else 'light_yellow' if g_rate < 80 else 'light_green'
                type_row.append(colored(f"{g_rate:.2f} %", g_color))
                row_raw.append(f"{g_rate:.2f} %")
            else:
                type_row.append("-")
                row_raw.append("-")
        
        success_rate = 100.0 * success / trials if trials > 0 else 0.0
        total_success += success
        total_trials += trials

        main_color = 'light_red' if success_rate < 50 else 'light_yellow' if success_rate < 80 else 'light_green'

        table_data.append([
            args.robot_name,
            args.dataset,
            object_name, 
            colored(f"{success}/{trials}", main_color, attrs=['bold']), 
            colored(f"{success_rate:.2f} %", main_color, attrs=['bold'])
        ] + type_row)
        table_data_raw.append([
            args.robot_name,
            args.dataset,
            object_name, 
            f"{success}/{trials}", 
            f"{success_rate:.2f} %"
        ] + row_raw)

    # Horizontal line
    table_data.append('')
    table_data_raw.append('')

    overall_success_rate = 100.0 * total_success / total_trials if total_trials > 0 else 0.0

    # Add total row
    total_type_row_raw = []
    total_type_row = []
    for g_type in ROBOTS_GRASPS_LABELS[args.robot_name].keys():
        g_success = type_total_result[g_type]['success']
        g_total = type_total_result[g_type]['total']
        if g_total > 0:
            g_rate = 100.0 * g_success / g_total if g_total > 0 else 0.0
            g_color = 'light_red' if g_rate < 50 else 'light_yellow' if g_rate < 80 else 'light_green'
            total_type_row.append(colored(f"{g_rate:.2f} %", g_color, attrs=['bold']))
            total_type_row_raw.append(f"{g_rate:.2f} %")
        else:
            total_type_row.append("-")
            total_type_row_raw.append("-")
    
    table_data_raw.append([
        args.robot_name.upper(),
        args.dataset.upper(),
        "TOTAL", 
        f"{total_success}/{total_trials}", 
        f"{overall_success_rate:.2f} %"
    ] + total_type_row_raw)
    table_data.append([
        colored(args.robot_name.upper(), 'cyan', attrs=['bold']),
        colored(args.dataset.upper(), 'cyan', attrs=['bold']),
        colored("TOTAL", 'cyan', attrs=['bold']), 
        colored(f"{total_success}/{total_trials}", 'cyan', attrs=['bold']), 
        colored(f"{overall_success_rate:.2f} %", 'cyan', attrs=['bold'])
    ] + total_type_row)

    headers = ["Robot", "Dataset", "Object Name", "Count", "Rate"] + list(ROBOTS_GRASPS_LABELS[args.robot_name].keys())
    output_lines.append(tabulate(table_data_raw, headers=headers, tablefmt="heavy_outline", stralign="left"))
    print(tabulate(table_data, headers=headers, tablefmt="heavy_outline", stralign="left"))

    # Compute Diversity
    diversity_pos, diversity_rot, diversity_joints = 0.0, 0.0, 0.0
    if len(all_success_q) > 1:
        all_success_q_concat = q_rot6d_to_q_euler(torch.cat(all_success_q, dim=0))
        diversity_pos = torch.std(all_success_q_concat[:, :3], dim=0).mean().item()
        diversity_rot = torch.std(all_success_q_concat[:, 3:6], dim=0).mean().item()
        diversity_joints = torch.std(all_success_q_concat[:, 6:], dim=0).mean().item()

    diversity_str = f"******** [{args.robot_name.upper()}] Overall Success Rate: {overall_success_rate:.2f} %   ---   Diversity (std): T={diversity_pos:.4f}, R={diversity_rot:.4f}, Q={diversity_joints:.4f}"

    output_lines.append(diversity_str)
    cprint(diversity_str, 'cyan', attrs=['bold'])
    
    # Write to log file
    log_file = os.path.join('logs_validation', f'{date_str}_results_{args.dataset}.txt')
    if not os.path.exists('logs_validation'):
        os.makedirs('logs_validation', exist_ok=False)
    with open(log_file, 'a') as f:
        f.write('\n'.join(output_lines) + '\n')