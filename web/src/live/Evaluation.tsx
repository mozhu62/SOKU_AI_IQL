import {useState} from 'react';
import {api,command,useQuery} from '../api';
import {Card,ConfirmButton,DataTable,Stat} from '../components/common';
import {Button} from '../components/ui/button';
import {number,percent,signed,type Row} from '../lib/utils';
import {roundColumns} from './Overview';

function Report({report}:{report:Row}){
  const [page,setPage]=useState(0),[current,setCurrent]=useState(report),[error,setError]=useState('');
  const move=async(next:number)=>{try{setCurrent(await api(`records/${report.id}?offset=${next*100}&limit=100`));setPage(next);setError('');}catch(e){setError(String(e));}};
  const s=current.summary||{};
  return <Card title={report.name} note={`模型 Step ${report.session?.step??'—'} · ${s.completed??0} 个完整小局`}><div className="stats-grid compact"><Stat title="平均伤害差" value={signed(s.mean_damage_difference,0)}/><Stat title="胜率" value={percent(s.win_rate)}/><Stat title="伤害差标准差" value={number(s.std_damage_difference,0)}/><Stat title="片段数量" value={number(s.fragments,0)}/></div>
    <DataTable rows={current.results||[]} columns={roundColumns}/><div className="inline-controls"><Button variant="secondary" disabled={!page} onClick={()=>void move(page-1)}>上一页</Button><span>第 {page+1} 页 / {current.total} 条</span><Button variant="secondary" disabled={(page+1)*100>=current.total} onClick={()=>void move(page+1)}>下一页</Button></div>
    {error&&<p className="error">{error}</p>}<details><summary>评估条件与模型指纹</summary><pre>{JSON.stringify({checkpoint:report.session?.checkpoint,sha256:report.session?.checkpoint_sha256,conditions:report.conditions},null,2)}</pre></details></Card>;
}

export function Evaluation({state}:{state:Row}){
  // 保留现有模型选择和双报告对比流程；实战不构造优化器，不修改训练模型。
  const files=useQuery('files',3000),[model,setModel]=useState(''),[rounds,setRounds]=useState(20),[difficulty,setDifficulty]=useState(''),[reports,setReports]=useState<Row[]>([]),[error,setError]=useState(''),[message,setMessage]=useState('');
  const known=(r:Row)=>{const value=r.conditions?.environment?.cpu_difficulty_label;return typeof value==='string'&&value.trim()&&!value.includes('未确认')&&r.conditions?.observed_modes?.length>0;};
  const same=reports.length===2&&reports.every(known)&&JSON.stringify(reports[0].conditions)===JSON.stringify(reports[1].conditions);
  const canLoad=state.connected&&!state.workbench_busy&&!state.control?.active&&state.state!=='initializing';
  const load=async()=>{const result=await command('evaluate',{model_id:model||null,rounds,difficulty:difficulty.trim()||state.runtime_config?.environment?.cpu_difficulty_label,confirm_discard:true});setMessage(result.message);};
  return <><div className="section-intro"><div><h1>固定模型实战评估</h1><p>先在同一人机难度下比较平均伤害差，再看胜率、样本量与波动。</p></div></div>
    <Card title="模型与评估设置" note="暂停后换模型 · 不更新权重 · 最大分类概率（argmax logits）"><div className="form-grid"><label className="field">评估模型<select value={model} onChange={e=>setModel(e.target.value)}><option value="">当前模型：{state.checkpoint||'配置中的模型'}</option>{(files.data?.models||[]).map((r:Row)=><option key={r.id} value={r.id}>{r.name}</option>)}</select></label><label className="field">完整小局数<input type="number" min={0} max={10000} value={rounds} onChange={e=>setRounds(+e.target.value)}/></label><label className="field">CPU 难度（按游戏实际设置）<input value={difficulty} onChange={e=>setDifficulty(e.target.value)} placeholder={state.runtime_config?.environment?.cpu_difficulty_label||'请确认游戏设置'}/></label></div>
      <div className="inline-controls"><ConfirmButton disabled={!canLoad} title="载入模型并新建评估会话" description="当前会话结束，未完小局作为片段保存。新模型载入后保持暂停，请点顶部继续再控制游戏。旧模型和训练 last.pt 都不会被修改。" onConfirm={load}>载入评估模型</ConfirmButton><span className="muted">完整小局 {number(state.summary?.completed,0)} / {state.summary?.target===0?'不限':number(state.summary?.target,0)}</span></div>
      {state.control?.active&&<p className="notice">正在控制游戏，先暂停再切换模型。点击本机浏览器可能已经触发失焦保护。</p>}{message&&<p role="status">{message}</p>}{files.error&&<p className="error">{files.error}</p>}
    </Card>
    <Card title="选择两次评估进行对比" note="只读已记录结果；不按训练奖励自动挑选或覆盖 best.pt"><DataTable rows={files.data?.evaluations||[]} columns={[{key:'name',title:'评估会话'},{key:'modified',title:'报告更新',render:v=>new Date(v*1000).toLocaleString()},{key:'id',title:'操作',render:id=><Button variant="ghost" onClick={()=>api(`records/${id}`).then(r=>{setReports(old=>[...old.slice(-1),r]);setError('');}).catch(e=>setError(String(e)))}>加入对比</Button>}]}/></Card>
    {error&&<p className="error">{error}</p>}{reports.length>0&&<><p className={same?'notice':'muted'}>{reports.length<2?'再选一个评估会话。':same?'已记录条件一致，可以对比；仍要检查样本量、缺帧和波动。':'CPU 难度未确认或条件不一致，只能并列查看，不能直接判定谁更强。'}</p><div className="two-columns">{reports.map((report,index)=><Report key={`${report.id}_${index}`} report={report}/>)}</div></>}
  </>;
}
