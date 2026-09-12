import {useEffect,useState} from 'react';
import {Activity,ChartNoAxesCombined,SlidersHorizontal,Play,Pause,Save,Square,CheckCheck} from 'lucide-react';
import {api,command,useStatus} from './api';
import {Button} from './components/ui/button';
import {ConfirmButton} from './components/common';
import {number} from './lib/utils';
import {Overview} from './views/Overview';
import {Diagnostics} from './views/Diagnostics';
import {Parameters} from './views/Parameters';

const tabs=[['overview','训练总览',ChartNoAxesCombined],['diagnostics','IQL 学习诊断',Activity],['parameters','参数与模型',SlidersHorizontal]] as const;
const labels:Record<string,string>={initializing:'准备数据',paused:'已暂停',training:'离线训练',validating:'固定集验证',stopped:'已停止',completed:'训练完成',error:'运行异常'};

export default function App(){
  useEffect(()=>{const context=(document as any).modelContext;if(!context?.registerTool)return;const lifetime=new AbortController();try{Promise.resolve(context.registerTool({name:'read_iql_training_status',description:'读取 IQL 状态、损失和训练速度，不修改模型或控制训练',inputSchema:{type:'object',properties:{},additionalProperties:false},annotations:{readOnlyHint:true},execute:()=>api('status')},{signal:lifetime.signal})).catch(()=>{});}catch{}return()=>lifetime.abort();},[]);
  const {state,connected}=useStatus(),[tab,setTab]=useState('overview'),[busy,setBusy]=useState(false),[message,setMessage]=useState('');
  const terminal=['stopped','completed','error'].includes(state.state), ready=connected&&state.ready&&!terminal;
  const act=async(name:string)=>{setBusy(true);try{const r=await command(name);setMessage(r.status==='queued'?'请求仍在排队，可稍后查看运行状态':r.message);}catch(e){setMessage(String(e));}finally{setBusy(false);}};
  if(state.ui_mode&&state.ui_mode!=='iql_training')return <main className="login-note">这是实战服务，请打开实战入口地址。</main>;
  return <div className="app-shell"><aside className="sidebar"><div className="brand"><span className="brand-mark">S</span><div><strong>SOKU / IQL</strong><small>离线训练工作台</small></div></div><nav aria-label="训练模块">{tabs.map(([id,label,Icon])=><button key={id} className={tab===id?'active':''} onClick={()=>setTab(id)}><Icon size={18}/><span>{label}</span></button>)}</nav><div className="sidebar-foot">本机 / SSH · {location.host}<small>BC 初始化 · Joint144</small></div></aside>
    <div className="workspace"><header className="topbar"><div className="run-status"><span className={`status-dot ${connected?'online':''}`}/><strong>{labels[state.state]||'连接中'}</strong><span>Step {number(state.step,0)} · 阶段 {state.stage??0}</span></div><div className="toolbar">
      <Button disabled={!ready||busy||state.state!=='paused'} onClick={()=>void act('resume')}><Play size={15}/>开始 / 继续</Button>
      <Button variant="secondary" disabled={!ready||busy||state.state==='paused'} onClick={()=>void act('pause')}><Pause size={15}/>暂停</Button>
      <Button variant="secondary" disabled={!ready||busy} onClick={()=>void act('save')}><Save size={15}/>保存版本</Button>
      <Button variant="secondary" disabled={!ready||busy||state.state!=='paused'} onClick={()=>void act('validate')}><CheckCheck size={15}/>验证</Button>
      <ConfirmButton disabled={!connected||terminal} title="停止并保存" description="等待当前完整更新结束后保存；数据准备期间停止只取消准备，不产生未初始化模型。" onConfirm={()=>command('stop')}><Square size={15}/>停止</ConfirmButton>
    </div></header><div className="connection-strip"><span>{connected?'已连接':'正在重连；浏览器断开不停止训练，也不会自动恢复暂停'}</span><span>原 BC Actor · 独立双 Q / V · 不接管游戏</span></div>
    {(message||state.error)&&<div className="feedback" role="status">{state.error||message}<button aria-label="关闭提示" onClick={()=>setMessage('')}>×</button></div>}
    <div className="runtime-message">{state.message||'等待服务状态'}</div><main>{tab==='overview'?<Overview state={state}/>:tab==='diagnostics'?<Diagnostics state={state}/>:<Parameters state={{...state,connected}}/>}</main>
    <footer><span>{state.output||'等待模型输出目录'}</span><span>有效训练帧 {number(state.samples,0)}（含重复采样） · 最近保存 Step {number(state.last_saved?.step,0)}</span></footer></div></div>;
}
