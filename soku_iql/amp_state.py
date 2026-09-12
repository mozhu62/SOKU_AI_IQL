"""CPU/非 AMP checkpoint 与 CUDA AMP 之间的缩放器恢复规则。"""
import logging


LOGGER = logging.getLogger(__name__)


def restore_grad_scaler(scaler, state, name):
    if not isinstance(state, dict):
        raise ValueError(f"{name} GradScaler 状态必须是字典，checkpoint 可能损坏")
    if not scaler.is_enabled():
        if state:
            LOGGER.warning("%s：当前未启用 AMP，不加载原 GradScaler；不属于相同精度模式的精确续训", name)
        return "disabled"
    if not state:
        # 禁用的 GradScaler 合法保存为 {}；CPU 迁移包没有可恢复的缩放历史。
        # 保留新 Learner 创建的默认缩放器，不伪造或继承来源模型的旧 AMP 历史。
        LOGGER.warning("%s：checkpoint 的 GradScaler 状态为空（CPU 迁移/未启用 AMP）；"
                       "使用新初始化的 CUDA GradScaler，模型和优化器仍正常恢复", name)
        return "initialized"
    # 非空但损坏的状态仍交给 PyTorch 严格校验，不吞掉其他恢复错误。
    scaler.load_state_dict(state)
    return "restored"
