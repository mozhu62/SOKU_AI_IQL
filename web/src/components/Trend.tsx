import {CartesianGrid,Legend,Line,LineChart,ResponsiveContainer,Tooltip,XAxis,YAxis} from 'recharts';
import {Card} from './common';
import type {Row} from '../api';

export function Trend({title,rows,series,note,numericSteps=false}:{title:string;rows:Row[];series:{key:string;name:string;color:string;connectNulls?:boolean}[];note?:string;numericSteps?:boolean}){
  const stages=[...new Set(rows.map(row=>row.stage??0))];
  // 不同配置阶段使用独立曲线，避免把改参前后的误差连接成同一实验。
  const data=rows.map(row=>{const item:Row={step:row.step};for(const s of series)item[`${s.key}_${row.stage??0}`]=row[s.key];return item;});
  return <Card title={title} note={note}>{rows.length?<div className="chart"><ResponsiveContainer width="100%" height="100%"><LineChart data={data} margin={{top:5,right:16,bottom:0,left:5}}>
    <CartesianGrid stroke="#28364d" strokeDasharray="3 3"/><XAxis dataKey="step" type={numericSteps?'number':'category'} domain={numericSteps?['dataMin','dataMax']:undefined} tick={{fill:'#95a5bc'}} minTickGap={40}/><YAxis width={62} tick={{fill:'#95a5bc'}} domain={['auto','auto']}/>
    <Tooltip contentStyle={{background:'#132239',border:'1px solid #405575'}} labelFormatter={value=>`更新步 ${value}`}/><Legend/>
    {series.flatMap(s=>stages.map(stage=><Line key={`${s.key}_${stage}`} dataKey={`${s.key}_${stage}`} name={`${s.name}${stages.length>1?` · 阶段 ${stage}`:''}`} stroke={s.color} strokeWidth={2} dot={rows.length<10?{r:3}:false} isAnimationActive={false} connectNulls={s.connectNulls??true}/>))}
  </LineChart></ResponsiveContainer></div>:<div className="empty">尚未记录；开始训练后显示真实数据</div>}</Card>;
}
