import {useState} from 'react';
import {DataTable} from './common';
import {number,percent} from '../lib/utils';

type Counts=number[]|Record<string,number>;

export function JointFrequency({catalog,counts,total}:{catalog?:string[];counts?:Counts;total?:number}){
  const [limit,setLimit]=useState(20);
  if(!counts||!catalog)return <p className="muted">未记录 Joint Action 分布。</p>;
  const rows=catalog.map((name,id)=>{
    // 统计既可能是数组，也可能是 JSON 字符串键字典；先区分类型再读取同一个动作 ID。
    const count=(Array.isArray(counts)?counts[id]:counts[String(id)])??0;
    return {id,name,count,rate:total?count/total:null};
  })
    .sort((a,b)=>b.count-a.count||a.id-b.id).slice(0,limit);
  return <><label className="inline-controls">展示范围 <select value={limit} onChange={e=>setLimit(Number(e.target.value))}>
    <option value={20}>Top 20</option><option value={50}>Top 50</option><option value={144}>全部 144</option>
  </select><span>{number(total,0)} 个有效样本</span></label>
    <DataTable rows={rows} columns={[{key:'name',title:'完整 Controller State'},{key:'id',title:'Joint ID'},
      {key:'count',title:'次数',render:v=>number(v,0)},{key:'rate',title:'占比',render:percent}]}/></>;
}
