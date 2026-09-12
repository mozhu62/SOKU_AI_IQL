import {useState} from 'react';
import {ChartNoAxesCombined,Swords,SlidersHorizontal,Trophy,Play,Pause,Save,Square,Wifi,WifiOff} from 'lucide-react';
import {command,useStatus} from '../api';
import {Button} from '../components/ui/button';
import {ConfirmButton} from '../components/common';
import {number} from '../lib/utils';
import {Overview} from './Overview';
import {Battle} from './Battle';
import {Evaluation} from './Evaluation';
import {Parameters} from './Parameters';
import {ModelToolbar} from './ModelToolbar';

// 复用原战斗工作台；BC 显示分类概率，不把 logits 标成 Q 或胜率。
const tabs=[['overview','实战总览',ChartNoAxesCombined],['battle','战斗与动作',Swords],['evaluation','实战评估',Trophy],['parameters','参数与记录',SlidersHorizontal]] as const;
const labels:Record<string,string>={initializing:'正在加载模型',paused:'等待 / 暂停',evaluating:'固定模型实战',stopped:'已停止',error:'运行异常'};
export default function LiveApp(){
  const {state,connected}=useStatus(),[tab,setTab]=useState('battle'),[message,setMessage]=useState(''),[busy,setBusy]=useState(false);
  const available=connected&&state.loaded&&!state.workbench_busy&&state.state!=='stopped'&&state.state!=='error';
  const act=async(name:string)=>{setBusy(true);try{const r=await command(name);setMessage(r.message||'请求已提交');}catch(e){setMessage(String(e));}finally{setBusy(false);}};
  const summary=state.summary||{};
  if(state.ui_mode&&state.ui_mode!=='iql_live')return <main className="login-note"><h1>这是离线训练服务</h1><p>请在 Windows 用 scripts/play.py 启动实战，再打开它打印的地址。离线训练模型不会从网页直接接管游戏。</p></main>;
  return <div className="app-shell"><aside className="sidebar"><div className="brand"><span className="brand-mark">S</span><div><strong>SOKU / IQL</strong><small>实战工作台</small></div></div><nav aria-label="实战模块">{tabs.map(([id,label,Icon])=><button key={id} className={id===tab?'active':''} onClick={()=>setTab(id)}><Icon size={18}/><span>{label}</span></button>)}</nav><div className="sidebar-foot"><span className="local-dot"/>本机 · {location.host}<small>固定权重 / Joint144 分类</small></div></aside>
    <div className="workspace"><header className="topbar"><ModelToolbar state={state} connected={connected}/><div className="run-status"><span className={`status-dot ${connected?'online':''}`}/><strong>{state.workbench_operation==='browse_model'?'正在选择模型':state.workbench_busy?'处理控制请求':labels[state.state]||'连接中'}</strong><span className="muted">模型 Step {number(state.step,0)} · {state.device||'—'}</span></div><div className="toolbar">
      <Button disabled={!available||busy||state.control?.active} onClick={()=>void act('resume')}><Play size={15}/>继续</Button>
      <Button variant="secondary" disabled={!available||busy} onClick={()=>void act('pause')}><Pause size={15}/>暂停</Button>
      <Button variant="secondary" disabled={!available||busy} onClick={()=>void act('save')}><Save size={15}/>保存报告</Button>
      <Button variant="secondary" onClick={()=>setTab('evaluation')}><Trophy size={15}/>评估</Button>
      <ConfirmButton disabled={!connected||busy} title="停止实战并保存报告" description="释放按键，关闭文件选择窗口，未完小局保存为片段；不修改任何模型。网页保留，可以重新加载模型。" onConfirm={()=>command('stop')}><Square size={14}/>停止</ConfirmButton>
    </div></header><div className="connection-strip">{connected?<Wifi size={14}/>:<WifiOff size={14}/>}<span>{connected?'已连接本机实战服务':'连接中断，正在重连；网页断开不会自动停止控制'}</span><span>F10 紧急暂停 · 点击浏览器会失焦，继续时切回游戏</span></div>
    {(message||state.workbench_error||state.error)&&<div className="feedback" role="status">{state.workbench_error||(state.error?state.message:message)}<button onClick={()=>setMessage('')} aria-label="关闭提示">×</button></div>}
    {state.message&&<div className="runtime-message">{state.message}</div>}
    <main>{tab==='overview'?<Overview state={state}/>:tab==='battle'?<Battle state={state}/>:tab==='evaluation'?<Evaluation state={{...state,connected}}/>:<Parameters state={{...state,connected}}/>}</main>
    <footer><span>{state.checkpoint||'等待模型路径'}</span><span>完整小局 {number(summary.completed,0)} · 片段 {number(summary.fragments,0)} · 不训练、不覆盖 last.pt</span></footer></div></div>;
}
