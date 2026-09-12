import {Card,Stat} from '../components/common';
import {number,type Row} from '../lib/utils';

export function TemporalStatus({state}:{state:Row}){
  if(state.temporal?.mode!=='tcn')return <p className="notice">当前 checkpoint 不是兼容的 TCN 网络，已拒绝进入实战控制。</p>;
  const capture=state.temporal_capture,prediction=state.prediction;
  if(!capture)return <p className="notice">尚未收到真实帧队列状态，请确认已重启推理服务。</p>;
  const required=capture.required_frames;
  return <Card title={`TCN${required} · 真实连续观测`} note={capture.connected?`LiveFrames.v1 队列已连接；历史分支读取 ${required} 帧 228D 状态`:'等待新版 DLL 的 LiveFrames.v1 队列；当前不会使用短历史发键'}>
    <div className="stats-grid compact">
      <Stat title="当前观测窗口" value={`${number(capture.buffer_frames,0)} / ${required}`} note={capture.ready?`游戏帧 ${capture.first_frame} — ${capture.last_frame}`:`积累中，满 ${required} 帧才执行模型推理`}/>
      <Stat title="最近一次推理使用" value={prediction?.context_frames_used!=null?`${prediction.context_frames_used} / ${required}`:'尚未推理'} note={prediction?.context_first_frame!=null?`游戏帧 ${prediction.context_first_frame} — ${prediction.observation_frame}；不代表当前缓存`:'不重复旧帧、不用零填充凑满窗口'}/>
      <Stat title="窗口重置" value={number(state.temporal_resets,0)} note={state.temporal_reset_reason||'尚无重置记录'}/>
      <Stat title="丢失采集槽" value={number(capture.dropped_slots,0)} note={`累计接收 ${number(capture.received_slots,0)} 个槽；队列容量 ${capture.capacity}`}/>
    </div>
    <p className="notice">暂停或失焦只停止发键，仍采集真实历史。推理稍慢时补收队列中的帧；换局或真正丢帧时重新积累，未满 {required} 帧不发模型动作。完整窗口不等于每秒一定完成 60 次推理。</p>
  </Card>;
}
