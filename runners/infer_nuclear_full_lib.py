import argparse

REQUIRED_SINGLE_AGENT_EXACT_KEYS = (
    "dino_wrapper.query_embed.weight",
    "eomt_head.class_head.weight",
)

REQUIRED_SINGLE_AGENT_PREFIXES = (
    "dino_wrapper.seg_blocks.",
    "dino_wrapper.dino.blocks.",
    "pose_score_net.",
)


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Nuclear workpiece full pipeline inference"
    )
    parser.add_argument("--nuclear_data_path", type=str, required=True)
    parser.add_argument(
        "--seg_ckpt",
        "--seg_ckpt_path",
        dest="seg_ckpt",
        type=str,
        required=True,
        help="Single-agent segmentation+pose checkpoint",
    )
    parser.add_argument(
        "--energy_ckpt",
        "--energy_ckpt_path",
        dest="energy_ckpt",
        type=str,
        default=None,
        help="Optional EnergyNet checkpoint for pose hypothesis ranking",
    )
    parser.add_argument(
        "--scale_ckpt",
        "--scale_ckpt_path",
        dest="scale_ckpt",
        type=str,
        default=None,
        help="Optional ScaleNet checkpoint for object size prediction",
    )
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--output_dir", type=str, default="./results/full_pipeline")
    parser.add_argument("--num_vis", type=int, default=5)
    parser.add_argument("--score_threshold", type=float, default=0.5)
    parser.add_argument(
        "--repeat_num",
        type=int,
        default=10,
        help="Number of pose hypothesis samples (more = more accurate but slower)",
    )
    parser.add_argument("--num_points", type=int, default=1024)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--device", type=str, default="cuda")
    return parser


def get_posenet_class():
    from networks.posenet_agent import PoseNet

    return PoseNet


def get_dbscan_class():
    try:
        from sklearn.cluster import DBSCAN
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "DBSCAN clustering requires scikit-learn at runtime. "
            "Install 'scikit-learn' or disable clustering before pose aggregation."
        ) from exc
    return DBSCAN


def get_torch_module():
    import torch

    return torch


def get_numpy_module():
    import numpy as np

    return np


def get_pose_runtime_deps():
    from networks.reward import sort_poses_by_energy
    from utils.misc import average_quaternion_batch
    from utils.metrics import get_rot_matrix
    from utils.transforms import matrix_to_quaternion, quaternion_to_matrix

    return (
        sort_poses_by_energy,
        average_quaternion_batch,
        get_rot_matrix,
        matrix_to_quaternion,
        quaternion_to_matrix,
    )


def validate_single_agent_seg_state_dict(state_dict):
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint model_state_dict must be a dict.")

    missing = [key for key in REQUIRED_SINGLE_AGENT_EXACT_KEYS if key not in state_dict]
    missing.extend(
        f"{prefix}*"
        for prefix in REQUIRED_SINGLE_AGENT_PREFIXES
        if not any(key.startswith(prefix) for key in state_dict)
    )
    if missing:
        missing_keys = ", ".join(missing)
        raise ValueError(
            "Checkpoint is not a valid single-agent nuclear segmentation checkpoint. "
            f"Missing required keys: {missing_keys}. "
            "Legacy segmentation checkpoints without integrated pose weights are not supported."
        )
    return state_dict


def _load_model_state_dict(checkpoint, ckpt_path):
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(
            f"Checkpoint '{ckpt_path}' is missing a 'model_state_dict' payload."
        )
    state_dict = checkpoint["model_state_dict"]
    if not isinstance(state_dict, dict):
        raise ValueError(
            f"Checkpoint '{ckpt_path}' has a non-dict 'model_state_dict' payload."
        )
    return state_dict


def _get_model_parameter_and_buffer_keys(model):
    parameter_keys = set()
    buffer_keys = set()

    named_parameters = getattr(model, "named_parameters", None)
    if callable(named_parameters):
        parameter_keys = {name for name, _ in named_parameters()}

    named_buffers = getattr(model, "named_buffers", None)
    if callable(named_buffers):
        buffer_keys = {name for name, _ in named_buffers()}

    return parameter_keys, buffer_keys


def _format_key_list(keys):
    if not keys:
        return "[]"
    return "[" + ", ".join(sorted(keys)) + "]"


def load_main_agent_checkpoint(agent, seg_ckpt_path):
    torch = get_torch_module()
    print(f"Loading single-agent seg checkpoint: {seg_ckpt_path}")
    checkpoint = torch.load(seg_ckpt_path, map_location="cpu")
    state_dict = validate_single_agent_seg_state_dict(
        _load_model_state_dict(checkpoint, seg_ckpt_path)
    )
    try:
        agent.net.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise ValueError(
            "Single-agent segmentation checkpoint could not be loaded into the main "
            f"runtime agent from '{seg_ckpt_path}': {exc}"
        ) from exc


def load_model_only(agent, ckpt_path, name):
    torch = get_torch_module()
    print(f"Loading {name} checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = _load_model_state_dict(checkpoint, ckpt_path)
    parameter_keys, buffer_keys = _get_model_parameter_and_buffer_keys(agent.net)
    if not parameter_keys and not buffer_keys:
        raise ValueError(
            f"{name} checkpoint '{ckpt_path}' cannot be validated against the "
            "target model because it does not expose named parameters or buffers."
        )

    checkpoint_keys = set(state_dict)
    missing_parameter_keys = sorted(parameter_keys - checkpoint_keys)
    unexpected_keys = sorted(checkpoint_keys - (parameter_keys | buffer_keys))
    if missing_parameter_keys or unexpected_keys:
        details = []
        if missing_parameter_keys:
            details.append(
                "missing parameter keys " + _format_key_list(missing_parameter_keys)
            )
        if unexpected_keys:
            details.append("unexpected keys " + _format_key_list(unexpected_keys))
        raise ValueError(
            f"{name} checkpoint '{ckpt_path}' is incompatible with the target model: "
            + "; ".join(details)
            + "."
        )
    try:
        load_result = agent.net.load_state_dict(state_dict, strict=False)
    except RuntimeError as exc:
        raise ValueError(
            f"{name} checkpoint could not be loaded from '{ckpt_path}': {exc}"
        ) from exc
    missing_keys = list(getattr(load_result, "missing_keys", []))
    unexpected_keys = list(getattr(load_result, "unexpected_keys", []))
    missing_parameter_keys = sorted(set(missing_keys).intersection(parameter_keys))
    fatal_unexpected_keys = sorted(
        key for key in unexpected_keys if key not in buffer_keys
    )
    if missing_parameter_keys or fatal_unexpected_keys:
        details = []
        if missing_parameter_keys:
            details.append(
                "missing parameter keys " + _format_key_list(missing_parameter_keys)
            )
        if fatal_unexpected_keys:
            details.append("unexpected keys " + _format_key_list(fatal_unexpected_keys))
        raise ValueError(
            f"{name} checkpoint '{ckpt_path}' is incompatible with the target model: "
            + "; ".join(details)
            + "."
        )
    tolerated_missing_buffers = sorted(
        key for key in missing_keys if key in buffer_keys
    )
    tolerated_unexpected_buffers = sorted(
        key for key in unexpected_keys if key in buffer_keys
    )
    if tolerated_missing_buffers or tolerated_unexpected_buffers:
        print(
            f"Loaded {name} checkpoint '{ckpt_path}' with non-fatal key mismatches: "
            f"missing={_format_key_list(tolerated_missing_buffers)}, "
            f"unexpected={_format_key_list(tolerated_unexpected_buffers)}"
        )


def build_cfg(args, agent_type="score", enable_segmentation=False):
    torch = get_torch_module()

    class Cfg:
        pass

    cfg = Cfg()
    cfg.device = args.device if torch.cuda.is_available() else "cpu"
    cfg.dino = "pointwise"
    cfg.pts_encoder = "pointnet2"
    cfg.agent_type = agent_type
    cfg.pose_mode = "rot_matrix"
    cfg.regression_head = "Rx_Ry_and_T"
    cfg.sde_mode = "ve"
    cfg.num_points = args.num_points
    cfg.img_size = args.img_size
    cfg.pointnet2_params = "light"
    cfg.parallel = False
    cfg.is_train = False
    cfg.eval = False
    cfg.pred = False
    cfg.use_pretrain = False
    cfg.log_dir = "infer_nuclear"
    cfg.ema_rate = 0.999
    cfg.lr = 1e-4
    cfg.lr_decay = 0.99
    cfg.optimizer = "Adam"
    cfg.warmup = 50
    cfg.grad_clip = 1.0
    cfg.sampling_steps = 500
    cfg.sampler_mode = ["ode"]
    cfg.energy_mode = "IP"
    cfg.s_theta_mode = "score"
    cfg.norm_energy = "identical"
    cfg.scale_embedding = 180
    cfg.eval_repeat_num = 50
    cfg.repeat_num = args.repeat_num
    cfg.num_gpu = 1
    cfg.scale_batch_size = 64
    cfg.save_video = False
    cfg.retain_ratio = 0.4
    cfg.clustering = 1
    cfg.clustering_eps = 0.05
    cfg.clustering_minpts = 0.1667
    cfg.enable_segmentation = enable_segmentation
    cfg.num_queries = 50
    cfg.query_inject_layer = -4
    cfg.num_object_classes = 6
    cfg.unfreeze_dino_last_n = 4
    cfg.seg_loss_weight = 1.0
    cfg.cls_loss_weight = 2.0
    return cfg


def init_pipeline_agents(args):
    PoseNet = get_posenet_class()

    main_cfg = build_cfg(args, agent_type="score", enable_segmentation=True)
    print("Building main single-agent segmentation+pose runtime agent...")
    main_agent = PoseNet(main_cfg)
    load_main_agent_checkpoint(main_agent, args.seg_ckpt)
    main_agent.net.eval()

    energy_agent = None
    if args.energy_ckpt:
        energy_cfg = build_cfg(args, agent_type="energy", enable_segmentation=False)
        print("Building pure GenPose2 energy agent...")
        energy_agent = PoseNet(energy_cfg)
        load_model_only(energy_agent, args.energy_ckpt, "energy")
        energy_agent.net.eval()

    scale_agent = None
    if args.scale_ckpt:
        scale_cfg = build_cfg(args, agent_type="scale", enable_segmentation=False)
        print("Building pure GenPose2 scale agent...")
        scale_agent = PoseNet(scale_cfg)
        load_model_only(scale_agent, args.scale_ckpt, "scale")
        scale_agent.net.eval()

    return main_cfg, main_agent, energy_agent, scale_agent


def build_instance_batch(pt_data, device):
    return {
        "pts": pt_data["pts"].unsqueeze(0).to(device),
        "pcl_in": pt_data["pcl_in"].unsqueeze(0).to(device),
        "roi_rgb": pt_data["roi_rgb"].unsqueeze(0).to(device),
        "roi_xs": pt_data["roi_xs"].unsqueeze(0).to(device),
        "roi_ys": pt_data["roi_ys"].unsqueeze(0).to(device),
        "pts_center": pt_data["pts_center"].unsqueeze(0).to(device),
        "zero_mean_pts": pt_data["zero_mean_pts"].unsqueeze(0).to(device),
    }


def aggregate_pose(cfg, pred_pose, pred_energy=None):
    torch = get_torch_module()
    np = get_numpy_module()
    (
        sort_poses_by_energy,
        average_quaternion_batch,
        get_rot_matrix,
        matrix_to_quaternion,
        quaternion_to_matrix,
    ) = get_pose_runtime_deps()

    bs, repeat_num, _ = pred_pose.shape
    if pred_energy is None:
        good_pose = pred_pose
    else:
        sorted_pose, _ = sort_poses_by_energy(pred_pose, pred_energy)
        retain_num = max(1, int(round(repeat_num * cfg.retain_ratio)))
        good_pose = sorted_pose[:, :retain_num, :]

    retain_num = good_pose.shape[1]
    rot_matrix = get_rot_matrix(
        good_pose[:, :, :-3].reshape(bs * retain_num, -1),
        cfg.pose_mode,
    )
    quat_wxyz = matrix_to_quaternion(rot_matrix).reshape(bs, retain_num, -1)
    aggregated_quat_wxyz = average_quaternion_batch(quat_wxyz)

    if getattr(cfg, "clustering", 0):
        DBSCAN = get_dbscan_class()
        min_samples = max(1, int(round(cfg.clustering_minpts * retain_num)))
        for batch_idx in range(bs):
            pairwise_distance = 1 - torch.sum(
                quat_wxyz[batch_idx].unsqueeze(0)
                * quat_wxyz[batch_idx].unsqueeze(1),
                dim=2,
            ) ** 2
            dbscan = DBSCAN(
                eps=cfg.clustering_eps,
                min_samples=min_samples,
            ).fit(pairwise_distance.cpu().numpy())
            labels = dbscan.labels_
            if np.any(np.asarray(labels) >= 0):
                bins = np.bincount(np.asarray(labels)[np.asarray(labels) >= 0])
                best_label = np.argmax(bins)
                aggregated_quat_wxyz[batch_idx] = average_quaternion_batch(
                    quat_wxyz[batch_idx, labels == best_label].unsqueeze(0)
                )[0]

    aggregated_trans = torch.mean(good_pose[:, :, -3:], dim=1)
    aggregated_pose = torch.zeros(bs, 4, 4, device=pred_pose.device)
    aggregated_pose[:, 3, 3] = 1.0
    aggregated_pose[:, :3, :3] = quaternion_to_matrix(aggregated_quat_wxyz)
    aggregated_pose[:, :3, 3] = aggregated_trans
    return aggregated_pose


def estimate_size_from_geometry(points, pose):
    torch = get_torch_module()
    rotation = pose[:, :3, :3]
    translation = pose[:, :3, 3]
    obj_points = points - translation.unsqueeze(1)
    obj_points = torch.bmm(rotation.transpose(1, 2), obj_points.transpose(1, 2))
    obj_points = obj_points.transpose(1, 2)
    bbox_length, _ = torch.max(torch.abs(obj_points), dim=1)
    return bbox_length * 2.0


def infer_pose_and_size(
    main_agent, energy_agent, scale_agent, cfg, pt_data, device, repeat_num
):
    data = build_instance_batch(pt_data, device)

    pred_pose, _ = main_agent.pred_func(data, repeat_num=repeat_num)
    pred_energy = None
    if energy_agent is not None:
        pred_energy = energy_agent.get_energy(
            data=data,
            pose_samples=pred_pose,
            T=None,
            mode="test",
            extract_feature=True,
        )

    aggregated_pose = aggregate_pose(cfg, pred_pose, pred_energy)
    final_pose = aggregated_pose.clone()

    if scale_agent is not None:
        scale_input = {
            "pts_feat": data["pts_feat"],
            "rgb_feat": data["rgb_feat"],
            "axes": aggregated_pose[:, :3, :3],
        }
        cal_mat, pred_size = scale_agent.pred_scale_func(scale_input)
        final_pose[:, :3, :3] = cal_mat
        size_source = "scale_net"
    else:
        pred_size = estimate_size_from_geometry(data["pcl_in"], aggregated_pose)
        size_source = "geometry"

    return {
        "R": final_pose[0, :3, :3].detach().cpu().numpy(),
        "t": final_pose[0, :3, 3].detach().cpu().numpy(),
        "size": pred_size[0].detach().cpu().numpy(),
        "size_source": size_source,
    }
