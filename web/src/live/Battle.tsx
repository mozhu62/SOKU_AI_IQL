import {useState} from 'react';
import {Card,DataTable,Stat} from '../components/common';
import {JointFrequency} from '../components/JointActions';
import {number,percent,type Row} from '../lib/utils';
import {ResourceInputs} from './ResourceInputs';

export const directions=['左下','下','右下','左','无方向','右','左上','上','右上'];
export const buttonNames=['体术 A','DASH D','轻弹幕 B','重弹幕 C'];
export const buttonKeys=['melee','dash','light_projectile','heavy_projectile'];

export function Battle({state}:{state:Row}){
  const p=state.prediction||{},game=state.game||{},round=state.current_round||{},summary=state.summary||{};
  const [limit,setLimit]=useState(20);
  const jointRows=(state.action_catalog||[]).map((name:string,id:number)=>({
    id,name,logit:p.joint_logits?.[id],probability:p.joint_probabilities?.[id],selected:p.joint_action_id===id,
    chosen:summary.joint_counts?.[id]??0,current:round.joint_counts?.[id]??0,readback:round.readback_joint_counts?.[id]??0,
  })).sort((a:Row,b:Row)=>(b.logit??-Infinity)-(a.logit??-Infinity)||a.id-b.id).slice(0,limit);
  const actual=Object.entries(round.actual_actions||{}).map(([id,count])=>({id,count})).sort((a,b)=>Number(b.count)-Number(a.count));
  const keyboard=state.runtime_config?.keyboard||{};
  const pressed=state.control?.pressed_keys?.map((key:string)=>key.toUpperCase()) as string[]|undefined;
  // 模型要求、当前已发送按键和 DLL 回读不能混为一项；回读是较新的游戏帧，不是命中确认。
  const buttonRows=buttonKeys.map((key,index)=>({
    name:buttonNames[index],key:keyboard[key],
    model:p.buttons?.[index]===undefined?null:Boolean(p.buttons[index]),
    sent:state.control?.pressed_inputs?state.control.pressed_inputs.includes(key):null,
    readback:game.readback?.[1]?.[index]===undefined?null:Boolean(game.readback[1][index]),
  }));
  const held=(value:unknown)=>value===true?'按下':value===false?'松开':'未记录';
  return <><div className="section-intro"><div><h1>战斗与策略输出</h1><p>训练来源 {state.algorithm?.toUpperCase()||'未加载'} · 单帧完整按键 · 确定性 argmax · 概率不是胜率或命中率。</p></div></div>
    <div className="stats-grid">
      <Stat title="最近模型选择" value={p.action??'等待决策'} note={p.direction?`九宫格 ${p.direction} · ${directions[p.direction-1]} · Joint ID ${p.joint_action_id}`:'未收到网络输出'}/>
      <Stat title="所选动作概率" value={percent(p.selected_probability)} note={`归一化熵 ${number(p.normalized_entropy,3)} · 不随机采样`}/>
      <Stat title="当前实际发送按键" value={pressed?.join(' + ')||'无'} note={p.execution_reason??'加载后点击继续，切回游戏'}/>
      <Stat title="游戏实际动作 / 动作帧" value={`${number(game.actual_action,0)} / ${number(game.action_frame,0)}`} note={`回读方向 ${game.readback?.[0]??'未记录'} · 游戏帧 ${game.frame??'未记录'}`}/>
    </div>
    <Card title="ABCD 按键核对（模型不输出卡牌）" note={`最近模型观测：小局 ${p.observation_round??'未记录'} / 帧 ${p.observation_frame??'未记录'}；暂停后保留最近输出，不代表仍在按键`}>
      <DataTable rows={buttonRows} columns={[{key:'name',title:'游戏输入'},{key:'key',title:'键盘映射'},{key:'model',title:'模型要求',render:held},{key:'sent',title:'当前发送状态',render:held},{key:'readback',title:'游戏实际回读',render:held}]}/>
      <p className="muted">按键成功发送不保证招式成功释放。受击、硬直、距离及取消条件可能使按键无法出招；状态见游戏 actionId 与回读。F10 或失焦会松开当前按键。</p>
    </Card>
    <Card title="Joint144 原始输出与概率排序" note={`上一帧实际输入 ${p.previous_joint_action_id===144?'START':state.action_catalog?.[p.previous_joint_action_id]??'未记录'}；上一帧方向持续量 ${number(p.previous_action_duration,3)}（clip 到 60 后 /60）`}>
      <label className="inline-controls">展示范围 <select value={limit} onChange={e=>setLimit(Number(e.target.value))}><option value={20}>Top 20</option><option value={50}>Top 50</option><option value={144}>全部 144</option></select></label>
      <DataTable rows={jointRows} columns={[{key:'name',title:'完整 Controller State'},{key:'id',title:'Joint ID'},{key:'logit',title:'原始 logit',render:v=>number(v,5)},{key:'probability',title:'策略概率',render:percent},{key:'selected',title:'模型选择',render:v=>v?'已选择':'—'},{key:'chosen',title:'会话发送数',render:v=>number(v,0)},{key:'current',title:'本局发送数',render:v=>number(v,0)},{key:'readback',title:'本局回读帧数',render:v=>number(v,0)}]}/>
    </Card>
    <div className="two-columns"><Card title="本局模型指令 Top-N"><JointFrequency catalog={state.action_catalog} counts={round.joint_counts} total={round.decisions}/></Card><Card title="本局实际控制回读 Top-N"><JointFrequency catalog={state.action_catalog} counts={round.readback_joint_counts} total={round.observed_frames}/></Card></div>
    <Card title="实际动作进入记录" note="来自游戏 actionId/动作帧变化；不把攻击键输入等同于成功出招或命中"><DataTable rows={actual} columns={[{key:'id',title:'游戏 actionId'},{key:'count',title:'本局进入次数',render:v=>number(v,0)}]}/></Card>
    <ResourceInputs usesResources={state.uses_resources} inputs={p.resource_inputs}/>
    <Card title="原始连接诊断"><details><summary>展开游戏、控制与完整网络输出</summary><pre>{JSON.stringify({game,control:state.control,prediction:p},null,2)}</pre></details></Card>
  </>;
}
