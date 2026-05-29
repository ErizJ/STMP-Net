import datetime
import time
import torch
from datetime import timedelta
import os
import csv
from torch.cuda import amp
import numpy as np
import torch.distributed as dist
from torch.nn import functional as F
from scipy import stats
from timm.utils import AverageMeter  # accuracy
from loss import (SupConLoss, ImSupConLoss, Fidelity_Loss, Fidelity_Loss_distortion,
                  Multi_Fidelity_Loss, InfoNCE_loss, loss_quality, ranking_loss_multi,
                  ranking_loss, CrossEntropyLabelSmooth)
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from eval import cross_eval
from utils import save_checkpoint


def save_results_to_csv(config, results_list, timestamp, filename="training_results.csv"):
    """保存训练结果到CSV文件"""
    # 保存到 result 文件夹
    result_dir = os.path.join("results", "training", config.DATA.DATASET)
    os.makedirs(result_dir, exist_ok=True)
    
    # 在文件名中添加日期时间戳
    base_name = filename.rsplit('.', 1)[0]  # 去掉扩展名
    ext = filename.rsplit('.', 1)[1] if '.' in filename else 'csv'
    filename_with_date = f"{base_name}_{timestamp}.{ext}"
    
    csv_path = os.path.join(result_dir, filename_with_date)
    
    with open(csv_path, 'w', newline='') as f:
        fieldnames = ['epoch', 'train_srcc', 'train_loss', 'val_plcc', 'val_srcc', 'val_plcc_cross', 'val_srcc_cross', 'lr']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in results_list:
            writer.writerow(result)
    
    return csv_path

def stage1_train(config, model, data_loader, epochs, optimizer, lr_scheduler, logger):
    max_plcc, max_srcc, max_plcc_c, max_srcc_c = 0.0, 0.0, 0.0, 0.0
    loss_scaler = amp.GradScaler()
    loss_meter = AverageMeter()
    
    # 生成时间戳用于区分不同的训练运行
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    start_time = time.monotonic()
    logger.info("start training")
    logger.info(f"Training timestamp: {timestamp}")
    logger.info(f"config.ALPHA: {config.ALPHA}, config.BETA: {config.BETA}, config.GAMMA: {getattr(config, 'GAMMA', 1.0)}")
    logger.info(f"Branches: scene={config.scene}, dist={config.dist}, texture={getattr(config, 'texture', False)}, structure={getattr(config, 'structure', False)}")

    pred_scores_list, gt_scores_list = [], []
    train_srcc = 0.0
    all_results = []  # 保存所有epoch的结果
    
    # 检查是否启用三分支
    use_texture = getattr(config, 'texture', False)
    use_structure = getattr(config, 'structure', False)

    def _to_float(v):
        if isinstance(v, torch.Tensor):
            return float(v.detach().item())
        return float(v)
    
    for epoch in range(1, epochs + 1):
        logger.info(f"Epoch{epoch} training")
        loss_meter.reset()
        pred_scores_list, gt_scores_list = [], []  # 每个epoch重置
        lr_scheduler.step(epoch)
        model.train()
        for n_iter, batch_data in enumerate(data_loader):
            # 根据数据格式解包
            if len(batch_data) == 6:
                # 六元组: (img, gt_score, scene_num, texture_num, structure_num, distortion_num)
                img, gt_score, scene_num, texture_num, structure_num, distortion_num = batch_data
            elif len(batch_data) == 5:
                # 五元组: (img, gt_score, scene_num, texture_num, structure_num)
                img, gt_score, scene_num, texture_num, structure_num = batch_data
                distortion_num = texture_num  # 无 dist 标签时占位，dist 分支不应激活
            else:
                # 四元组: (img, gt_score, scene_num, distortion_num)
                img, gt_score, scene_num, distortion_num = batch_data
                texture_num = distortion_num
                structure_num = distortion_num
            
            optimizer.zero_grad()
            img = img.cuda(non_blocking=True)
            gt_score = gt_score.cuda(non_blocking=True)
            scene_num = scene_num.cuda(non_blocking=True)
            distortion_num = distortion_num.cuda(non_blocking=True)
            if use_texture:
                texture_num = texture_num.cuda(non_blocking=True)
            if use_structure:
                structure_num = structure_num.cuda(non_blocking=True)

            with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
                outputs = model(img)
                pred_score = outputs[0]
                logits_global = outputs[1]
                logits_local = outputs[2]
                logits_texture = outputs[3] if len(outputs) > 3 else None
                logits_structure = outputs[4] if len(outputs) > 4 else None
                
            global_loss, local_loss, texture_loss, structure_loss = 0.0, 0.0, 0.0, 0.0

            # Scene 损失
            if logits_global is not None:
                smooth_loss_global = CrossEntropyLabelSmooth(num_classes=config.DATA.SCENE_NUM_CLASSES)
                global_loss = smooth_loss_global(logits_global, scene_num)

            # Dist 损失: 用真实的 distortion_num 监督（6元组时有效）
            if logits_local is not None:
                smooth_loss_local = CrossEntropyLabelSmooth(num_classes=config.DATA.DIST_NUM_CLASSES)
                local_loss = smooth_loss_local(logits_local, distortion_num)

            # Texture 损失
            if logits_texture is not None and use_texture:
                smooth_loss_texture = CrossEntropyLabelSmooth(num_classes=config.DATA.TEXTURE_NUM_CLASSES)
                texture_loss = smooth_loss_texture(logits_texture, texture_num)

            # Structure 损失
            if logits_structure is not None and use_structure:
                smooth_loss_structure = CrossEntropyLabelSmooth(num_classes=config.DATA.STRUCTURE_NUM_CLASSES)
                structure_loss = smooth_loss_structure(logits_structure, structure_num)

            fidelity_loss = loss_quality(pred_score, gt_score)
            smoothl1_loss = torch.nn.SmoothL1Loss()(pred_score, gt_score)
            
            # 总损失
            # global_loss: 1.0, texture_loss: α, structure_loss: β, smoothl1_loss: γ, fidelity_loss: δ
            loss = global_loss + config.ALPHA * texture_loss + config.BETA * structure_loss + config.GAMMA * smoothl1_loss + config.DELTA * fidelity_loss

            # 检测 NaN loss，跳过该 batch
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"Epoch[{epoch}] Iteration[{n_iter + 1}] 检测到 NaN/Inf loss，跳过该 batch")
                optimizer.zero_grad()
                continue

            loss_scaler.scale(loss).backward()
            loss_scaler.step(optimizer)
            loss_scaler.update()
            model.invalidate_text_cache()
            loss_meter.update(loss.item(), img.shape[0])
            pred_scores_list.extend(np.atleast_1d(pred_score.detach().cpu().numpy()).tolist())
            gt_scores_list.extend(np.atleast_1d(gt_score.detach().cpu().numpy()).tolist())
            train_srcc, _ = stats.spearmanr(pred_scores_list, gt_scores_list)
            torch.cuda.synchronize()
            if n_iter % 10 == 0:
                logger.info(
                    f"Epoch[{epoch}] Iteration[{n_iter + 1}/{len(data_loader)}] "
                    f"Loss: {loss_meter.avg:.3f}, "
                    f"Global_Loss: {_to_float(global_loss):.3f}, "
                    f"Local_Loss: {_to_float(local_loss):.3f}, "
                    f"Texture_Loss: {_to_float(texture_loss):.3f}, "
                    f"Structure_Loss: {_to_float(structure_loss):.3f}, "
                    f"Fidelity_Loss: {_to_float(fidelity_loss):.3f} "
                    f"Smoothl1_loss:{_to_float(smoothl1_loss):.3f} "
                    f"Base Lr: {lr_scheduler._get_lr(epoch)[0]:.2e}, train_srcc: {train_srcc:.3f}"
                )

        # 每个epoch结束后记录结果
        epoch_result = {
            'epoch': epoch,
            'train_srcc': round(train_srcc, 4),
            'train_loss': round(loss_meter.avg, 4),
            'val_plcc': 0.0,
            'val_srcc': 0.0,
            'val_plcc_cross': 0.0,
            'val_srcc_cross': 0.0,
            'lr': lr_scheduler._get_lr(epoch)[0]
        }

        if epoch % 4 == 0:
            # 使用配置中的 TTA 设置
            use_tta = getattr(config.TEST, 'USE_TTA', False)
            val_plcc, val_srcc, val_plccc, val_srccc = cross_eval(config, model, logger, use_tta=use_tta, in_domain_only=True)
            logger.info(f"stage 1 validate:{val_plcc}, {val_srcc}, {val_plccc}, {val_srccc}")
            
            epoch_result['val_plcc'] = round(val_plcc, 4)
            epoch_result['val_srcc'] = round(val_srcc, 4)
            epoch_result['val_plcc_cross'] = round(val_plccc, 4)
            epoch_result['val_srcc_cross'] = round(val_srccc, 4)
            
            if val_plcc >= max_plcc:
                max_plcc = val_plcc
                max_srcc = val_srcc
                max_plcc_c = val_plccc
                max_srcc_c = val_srccc
                # 保存最优模型
                save_checkpoint(config, epoch, model, max_plcc, optimizer, lr_scheduler, loss_scaler, logger)
                logger.info(f"Saved best model at epoch {epoch} with PLCC: {max_plcc:.4f}")
            logger.info(f"stage 1 max:{max_plcc}, {max_srcc}, {max_plcc_c}, {max_srcc_c}")
        
        all_results.append(epoch_result)

    # 保存所有结果到CSV
    csv_path = save_results_to_csv(config, all_results, timestamp)
    logger.info(f"Training results saved to: {csv_path}")
    
    # 保存最终最佳结果摘要（文件名中添加时间戳）
    result_dir = os.path.join("results", "training", config.DATA.DATASET)
    os.makedirs(result_dir, exist_ok=True)
    summary_path = os.path.join(result_dir, f"best_results_{timestamp}.csv")
    
    with open(summary_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['dataset', 'best_plcc', 'best_srcc', 'best_plcc_cross', 'best_srcc_cross', 'seed', 'epochs'])
        writer.writeheader()
        
        writer.writerow({
            'dataset': config.DATA.DATASET,
            'best_plcc': round(max_plcc, 4),
            'best_srcc': round(max_srcc, 4),
            'best_plcc_cross': round(max_plcc_c, 4),
            'best_srcc_cross': round(max_srcc_c, 4),
            'seed': config.SEED,
            'epochs': epochs
        })
    logger.info(f"Best results summary saved to: {summary_path}")

    end_time = time.monotonic()
    total_time = timedelta(seconds=end_time - start_time)
    logger.info("Stage1 running time: {}".format(total_time))
    return max_plcc, max_srcc, max_plcc_c, max_srcc_c
