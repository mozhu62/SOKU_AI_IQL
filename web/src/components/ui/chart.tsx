import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ReferenceLine, Brush } from 'recharts';
import type { Row } from '../../lib/utils';
export function Chart({rows,series,zero=false}:{rows:Row[];series:{key:string;name:string;color:string}[];zero?:boolean}) {
 if (!rows.length) return <div className="empty">尚无记录 · 完成采样或更新后显示</div>;
 return <div className="chart"><ResponsiveContainer width="100%" height="100%"><LineChart data={rows} margin={{top:8,right:18,left:0,bottom:8}}>
 <CartesianGrid stroke="#253347" strokeDasharray="3 5"/><XAxis dataKey="index" stroke="#91a1b8" minTickGap={24}/><YAxis stroke="#91a1b8" width={60}/>
 <Tooltip contentStyle={{background:'#142035',border:'1px solid #42516b',borderRadius:8}} labelFormatter={v=>`记录 ${v}`}/><Legend/>
 {zero&&<ReferenceLine y={0} stroke="#8393aa"/>}{series.map(s=><Line key={s.key} dataKey={s.key} name={s.name} stroke={s.color} dot={false} connectNulls={false} isAnimationActive={false} strokeWidth={2}/>)}
 <Brush dataKey="index" height={18} stroke="#506584" fill="#111c2d"/>
 </LineChart></ResponsiveContainer></div>;
}
