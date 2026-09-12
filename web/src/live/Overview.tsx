import {useState} from 'react';
import {useQuery} from '../api';
import {Card,Stat,DataTable} from '../components/common';
import {Chart} from '../components/ui/chart';
import {mean,number,percent,signed,type Row} from '../lib/utils';
import {TemporalStatus} from './TemporalStatus';

const resultLabels:Record<string,string>={win:'胜',loss:'负',draw:'平',interrupted:'中断'};
export const roundColumns=[
  {key:'index',title:'记录',render:(v:any)=>number(v,0)},
  {key:'result',title:'结果',render:(v:string)=>resultLabels[v]||v},
  {key:'damage_dealt',title:'造成伤害',render:(v:any)=>number(v,0)},
  {key:'damage_taken',title:'受到伤害',render:(v:any)=>number(v,0)},
  {key:'damage_difference',title:'伤害差',render:(v:any)=>signed(v,0)},
  {key:'eligible',title:'完整',render:(v:any)=>v?'是':'片段'},
  {key:'missing_frames',title:'缺帧',render:(v:any)=>number(v,0)},
  {key:'notes',title:'说明',render:(v:any,r:Row)=>v?.join('；')||r.outcome_source||'—'},
];

export function Overview({state}:{state:Row}){
  const [limit,setLimit]=useState(100),[window,setWindow]=useState(20);
  const history=useQuery(`history?kind=rounds&limit=${limit}`,1000),raw:Row[]=history.data?.rows||[];
  const recent:number[]=[];
  // 复用 PPO 的逐局图和移动均值，片段断开；不混入离线 TD 损失或训练奖励。
  const rows=raw.map(row=>{if(row.eligible)recent.push(row.damage_difference);return {...row,
    damage:row.eligible?row.damage_difference:null,average:row.eligible?mean(recent.slice(-window)):null,
    dealt:row.eligible?row.damage_dealt:null,taken:row.eligible?row.damage_taken:null};});
  const summary=state.summary||{},current=state.current_round||{},game=state.game||{};
  return <><div className="section-intro"><div><h1>实战总览</h1><p>固定模型对局：伤害差优先，胜率辅助；控制质量单独看。</p></div><div className="inline-controls"><label>展示 <select value={limit} onChange={e=>setLimit(+e.target.value)}>{[20,50,100].map(n=><option key={n}>{n}</option>)}</select> 条</label><label>均值窗口 <input type="number" min={2} max={100} value={window} onChange={e=>setWindow(Math.max(2,Math.min(100,+e.target.value)))}/></label></div></div>
    <TemporalStatus state={state}/>
    <div className="stats-grid"><Stat title="完整小局 · 平均伤害差" value={signed(summary.mean_damage_difference,0)} note={`标准差 ${number(summary.std_damage_difference,0)}`} tone={summary.mean_damage_difference>=0?'positive':''}/><Stat title="胜率" value={percent(summary.win_rate)} note={`胜/负/平 ${number(summary.wins,0)}/${number(summary.losses,0)}/${number(summary.draws,0)}`}/><Stat title="完整小局 / 目标" value={`${number(summary.completed,0)} / ${summary.target===0?'不限':number(summary.target,0)}`} note={`不完整片段 ${number(summary.fragments,0)} 条，单列保留`}/><Stat title="推理 P50 / P95" value={`${number(summary.inference_p50_ms,1)} / ${number(summary.inference_p95_ms,1)} ms`} note="含状态整理、计算及设备同步"/></div>
    <div className="two-columns"><Card title="逐局伤害差" note="片段断开；拖动底部窗口查看"><Chart rows={rows} zero series={[{key:'damage',name:'小局伤害差',color:'#667f9e'},{key:'average',name:`${window} 局均值`,color:'#42d6b0'}]}/></Card><Card title="造成与受到伤害" note="横轴是记录序号，不是训练更新步"><Chart rows={rows} series={[{key:'dealt',name:'造成',color:'#42d6b0'},{key:'taken',name:'受到',color:'#f2858c'}]}/></Card></div>
    <div className="two-columns"><Card title="当前小局"><div className="stats-grid compact"><Stat title="造成 / 受到伤害" value={`${number(current.damage_dealt,0)} / ${number(current.damage_taken,0)}`}/><Stat title="双方 HP" value={game.in_battle?`${number(game.hp,0)} / ${number(game.opponent_hp,0)}`:'等待战斗'}/><Stat title="游戏帧" value={number(game.frame,0)}/><Stat title="当前模式" value={game.battle_mode_name||'未连接'} note={`${game.battle_submode_name||'—'} · mode=${game.battle_mode??'—'} / submode=${game.battle_submode??'—'}`}/></div></Card><Card title="控制质量" note="不能把漏帧或发键问题当成模型能力差"><DataTable rows={[{name:'已发送决策',value:summary.decisions},{name:'按键回读确认',value:state.acknowledged},{name:'未确认/被覆盖',value:state.unconfirmed},{name:'缺失游戏帧',value:state.missing_frames},{name:'过期决策丢弃',value:state.stale_predictions}]} columns={[{key:'name',title:'指标'},{key:'value',title:'数量',render:v=>number(v,0)}]}/></Card></div>
    {state.stale_predictions>0&&<p className="notice">有 {state.stale_predictions} 次推理后复核失效，旧动作未发送。{state.temporal?.mode==='tcn'?'真实观测历史保留并继续追读；真正丢帧或控制中断的局按片段记录。':'受影响小局按片段保留，不计正式胜率。'}</p>}
    {history.error&&<p className="error">{history.error}</p>}
    <Card title="最近小局" note="首次中途接管、暂停、失焦和明显缺帧都保留记录"><DataTable rows={raw.slice(-20).reverse()} columns={roundColumns}/></Card>
  </>;
}
