import {useHistory,type Row} from '../api';
import {Card,Stat} from '../components/common';
import {Trend} from '../components/Trend';
import {number,percent} from '../lib/utils';

export function Overview({state}:{state:Row}){
  const train=useHistory('train'), validation=useHistory('validation');
  const t=state.latest_train||{},v=state.latest_validation||{},cfg=state.config||{},times=state.timings||{};
  const keys=[['data_wait','等待 CPU 数据'],['optimization','搬运 / 更新 / 诊断'],['validation','离线验证'],['save','模型保存'],['paused','暂停等待']] as const;
  const total=keys.reduce((sum,[key])=>sum+(times[key]||0),0), warmup=(state.step??0)<(cfg.iql?.actor_warmup_steps??1000);
  return <><div className="section-intro"><div><h1>从 BC 策略继续学习回报</h1><p>离线拟合与实战能力分开判断；伤害差与胜率请在 Windows 实战工作台评估。</p></div><span className="run-badge">{state.device||'等待设备'}</span></div>
    <div className="stats-grid"><Stat title="IQL 更新进度" value={`${number(state.step,0)} / ${number(cfg.training?.total_steps,0)}`} note={`Actor 实际更新 ${number(state.actor_updates,0)} 次`}/><Stat title="策略阶段" value={warmup?'Q/V 预热':'联合训练'} note={warmup?`前 ${cfg.iql?.actor_warmup_steps??1000} 步 Actor 保持 BC 权重`:'Actor 使用优势加权 CE，非普通 BC'}/><Stat title="验证动作一致率" value={percent(v.top1)} note={`NLL ${number(v.nll)} · 样本 ${number(v.samples,0)}`}/><Stat title="有效训练帧 / 秒" value={number(state.speed?.samples_per_second,0)} note={`${number(state.speed?.steps_per_second,2)} step/s · 最近 ${state.speed?.window_steps??0} 步活动时间`}/></div>
    <progress className="run-progress" aria-label="训练进度" value={state.step||0} max={cfg.training?.total_steps||1}/>
    <div className="notice-list">{warmup&&<p className="notice">预热时 Actor loss 未记录、策略不变是正常行为；不是训练卡住。</p>}{t.critic_updated===false&&<p className="notice">最近一次 Q 优化器没有更新，查看梯度与 AMP 状态；不能将循环步数当成成功更新次数。</p>}{v.change_count===0&&<p className="notice">本次验证没有动作切换帧，切换准确率不适用，不能视为 0%。</p>}{(train.error||validation.error)&&<p className="error">{train.error||validation.error}</p>}</div>
    <div className="two-columns"><Trend title="策略模仿误差" note="NLL 仍按有效帧等权；不是胜率" rows={validation.rows} series={[{key:'nll',name:'验证 NLL',color:'#62d8b6'}]}/><Trend title="动作一致率与复制基线" note="验证为固定种子抽样，非全量数据集" rows={validation.rows} series={[{key:'top1',name:'Top-1',color:'#62d8b6'},{key:'top5',name:'Top-5',color:'#71aaff'},{key:'previous_action_baseline',name:'复制上一帧基线',color:'#bfa6eb'},{key:'change_top1',name:'切换帧 Top-1',color:'#efac7f',connectNulls:false}]}/></div>
    <div className="two-columns"><Trend title="IQL 三项训练损失" note="权重与量纲不同，不能直接横向比大小" rows={train.rows} series={[{key:'actor_loss',name:'Actor 加权 CE',color:'#62d8b6',connectNulls:false},{key:'q_loss',name:'双 Q TD MSE 和',color:'#71aaff'},{key:'value_loss',name:'V Expectile',color:'#d5a8ed'}]}/><Card title="时间花在哪里" note="当前进程累计；准备阶段另计"><div className="time-list">{keys.map(([key,label])=><div key={key}><div><span>{label}</span><span>{number(times[key],1)} s · {total&&times[key]!=null?percent(times[key]/total):'未记录'}</span></div><progress aria-label={label} max={Math.max(total,1)} value={times[key]||0}/></div>)}</div><p className="metric-note">数据准备 {number(times.preparation,1)} s。取数只统计实际等待；预取可与 GPU 重叠。暂停仅累计等待命令的时间，不包含手动验证或保存。</p></Card></div>
  </>;
}
