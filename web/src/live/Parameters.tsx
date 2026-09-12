import {useEffect,useState} from 'react';
import {command} from '../api';
import {Card,ConfirmButton,DataTable} from '../components/common';
import type {Row} from '../lib/utils';
import {buttonKeys,buttonNames} from './Battle';

const keyNames:Record<string,string>={up:'上',down:'下',left:'左',right:'右',...Object.fromEntries(buttonKeys.map((key,i)=>[key,buttonNames[i]]))};

export function Parameters({state}:{state:Row}){
  const config=state.runtime_config||{},signature=JSON.stringify(config)||'{}',[draft,setDraft]=useState<Row>({}),[message,setMessage]=useState('');
  useEffect(()=>{const cfg=JSON.parse(signature);setDraft({device:cfg.device,rounds:cfg.rounds,cpu_threads:cfg.cpu_threads,
    keyboard:cfg.keyboard||{},difficulty:cfg.environment?.cpu_difficulty_label,
    decision_interval_frames:cfg.environment?.decision_interval_frames,auto_restart:cfg.restart?.enabled});},[signature]);
  const canApply=state.connected&&!state.workbench_busy&&!state.control?.active&&state.state!=='initializing';
  const apply=async()=>{const result=await command('load',{...draft,confirm_discard:true});setMessage(result.message);};
  const game=state.game||{};
  return <><div className="section-intro"><div><h1>参数与记录</h1><p>沿用工作台配置流程：编辑值与运行值分开，暂停后应用并新建会话。</p></div></div>
    <Card title="实战参数" note="不更改训练损失、网络结构或模型权重；分类选择固定为 argmax，不采样"><div className="form-grid">
      <label className="field">推理设备<input value={draft.device??''} onChange={e=>setDraft({...draft,device:e.target.value})} placeholder="cpu / cuda / cuda:0"/></label>
      <label className="field">CPU 线程数<input type="number" min={1} max={64} value={draft.cpu_threads??2} onChange={e=>setDraft({...draft,cpu_threads:+e.target.value})}/></label>
      <label className="field">完整小局数（0 不限）<input type="number" min={0} max={10000} value={draft.rounds??20} onChange={e=>setDraft({...draft,rounds:+e.target.value})}/></label>
      <label className="field">决策间隔 / 游戏帧<input type="number" min={1} max={1} disabled value={draft.decision_interval_frames??1} onChange={e=>setDraft({...draft,decision_interval_frames:+e.target.value})}/><small>固定为 1；从 DLL 队列补收真实帧，满 32 帧才推理。实际推理频率受性能限制。</small></label>
      <label className="field">游戏内 CPU 难度<input value={draft.difficulty??''} onChange={e=>setDraft({...draft,difficulty:e.target.value})}/></label>
      <label className="field">整场结束自动续局<select value={draft.auto_restart?'on':'off'} onChange={e=>setDraft({...draft,auto_restart:e.target.value==='on'})}><option value="on">连续按菜单确认键</option><option value="off">关闭，手动续局</option></select></label>
    </div></Card>
    <Card title="实际按键映射" note="WASD + J/K/I/L，须与游戏设置一致；本模型不输出切卡或用卡"><div className="table-scroll"><table><thead><tr><th>输入</th><th>编辑值</th><th>运行值</th></tr></thead><tbody>{Object.entries(keyNames).map(([key,name])=><tr key={key}><td>{name}</td><td><input aria-label={`${name} 按键`} maxLength={16} value={draft.keyboard?.[key]??''} onChange={e=>setDraft({...draft,keyboard:{...draft.keyboard,[key]:e.target.value}})}/></td><td>{config.keyboard?.[key]??'—'}</td></tr>)}</tbody></table></div>
      <div className="inline-controls parameter-actions"><ConfirmButton disabled={!canApply} title="应用设置并重新加载当前模型" description="结束旧评估会话并保存报告，未完小局作为片段。新会话保持暂停，确认后点击继续。" onConfirm={apply}>应用并重新加载</ConfirmButton></div>
      {message&&<p role="status">{message}</p>}<p className="muted">修改仅用于当前服务及会话记录；长期保存请编辑 configs/live_eval.yaml。F10 保留为紧急暂停。</p>
    </Card>
    <Card title="模式与连接诊断" note="主模式 2 = 人机；子模式 2 = REP，二者不得混淆"><DataTable rows={[
      {name:'主模式 battleMode',value:`${game.battle_mode??'未记录'} / ${game.battle_mode_name||'未记录'}`},
      {name:'子模式 battleSubMode',value:`${game.battle_submode??'未记录'} / ${game.battle_submode_name||'未记录'}`},
      {name:'inBattle / matchState',value:`${String(game.in_battle??'未记录')} / ${game.match_state??'未记录'}`},
      {name:'进程 ID / 游戏帧',value:`${game.pid??'未记录'} / ${game.frame??'未记录'}`},
      {name:'网络版本',value:state.network_version||'尚未加载'},
      {name:'时序结构',value:state.temporal?.mode==='tcn'?'TCN32：228D×32 → 256D':'尚未加载'},
      {name:'技能类型输入',value:state.uses_resources===true?'每方四槽 variant + mask，同帧配对':state.uses_resources===false?'旧模型，不使用':'尚未加载'},
      {name:'评估报告目录',value:state.summary?.report_directory||'尚未创建'},
    ]} columns={[{key:'name',title:'字段'},{key:'value',title:'运行值'}]}/></Card>
    <Card title="本次运行配置" note="编辑值不会在控制游戏时偷偷生效"><pre>{JSON.stringify(config,null,2)}</pre></Card>
  </>;
}
