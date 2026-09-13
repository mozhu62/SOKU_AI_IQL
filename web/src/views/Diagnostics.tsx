import {useHistory,useQuery,type Row} from '../api';
import {Card,DataTable,Stat} from '../components/common';
import {Trend} from '../components/Trend';
import {number,percent} from '../lib/utils';

export function Diagnostics({state}:{state:Row}){
  const train=useHistory('train'),val=useHistory('validation'),actions=useQuery('actions',5000), v=state.latest_validation||{},t=state.latest_train||{};
  const catalog:string[]=actions.data?.catalog||[],counts:number[]=actions.data?.counts||[],total=counts.reduce((a,b)=>a+b,0);
  const rows=catalog.map((name,id)=>({id,name,count:counts[id]??null,fraction:total?(counts[id]||0)/total:null})).sort((a,b)=>(b.count||0)-(a.count||0));
  return <><h1 className="page-heading">Actor / Q / V 学习诊断</h1><p className="notice">TD EV 衡量 min(Q1,Q2) 对单步目标 r + γV(s′) 的解释程度；终局不使用后继 V。不是蒙特卡洛回报 EV，与旧 N-step 指标不可直接比较。离线 NLL 最佳不等于实战最佳。</p>
    <div className="stats-grid"><Stat title="验证 TD MSE" value={number(v.td_mse)} note={`MAE ${number(v.td_mae)}`}/><Stat title="验证 TD EV" value={v.ev==null?'不适用 / 未记录':number(v.ev)} note="目标方差为零时不报告 EV"/><Stat title="优势权重均值 / 最大值" value={`${number(t.weight_mean)} / ${number(t.weight_max)}`} note={`平均优势 ${number(t.advantage_mean)}`}/><Stat title="本次验证有效帧" value={number(v.samples,0)} note={`其中切换帧 ${number(v.change_count,0)}`}/></div>
    <div className="two-columns"><Trend title="Q 的下一批预测质量" rows={val.rows} series={[{key:'td_mse',name:'TD MSE',color:'#71aaff'},{key:'td_mae',name:'TD MAE',color:'#62d8b6'}]}/><Trend title="验证 Q / V 数值" note="不是动作概率" rows={val.rows} series={[{key:'q_mean',name:'min Q 均值',color:'#71aaff'},{key:'v_mean',name:'V 均值',color:'#d5a8ed'}]}/><Trend title="模块梯度范数" note="裁剪前范数；缺失或非有限值不伪装为零" rows={train.rows} series={[{key:'actor_gradient_norm',name:'Actor',color:'#62d8b6',connectNulls:false},{key:'critic_gradient_norm',name:'Q1+Q2',color:'#71aaff',connectNulls:false},{key:'value_gradient_norm',name:'V',color:'#d5a8ed',connectNulls:false}]}/><Trend title="优势权重变化" rows={train.rows} series={[{key:'weight_mean',name:'均值',color:'#62d8b6'},{key:'weight_max',name:'最大值',color:'#efac7f'}]}/></div>
    <Card title="Actor Neutral 损失权重" note="类别加权与旧筛选互斥；Q/V、验证和时序输入不变">
      <p>类别加权：{t.actor_weighting_enabled==null?'未记录':t.actor_weighting_enabled?'已启用':'未启用'} · Neutral × {number(t.actor_neutral_weight)} · 其他动作 × 1。加权模式保留全部有效帧；全 Neutral 批次的类别权重会在归一化时抵消。</p>
      <div className="stats-grid"><Stat title="Neutral 筛选前 / 后" value={`${percent(t.actor_neutral_fraction_before)} / ${percent(t.actor_neutral_fraction_after)}`} note={`上限 ${percent(t.actor_neutral_max_fraction)} · ${t.actor_sampling_enabled==null?'未记录':t.actor_sampling_enabled?'已启用':'未启用'}`}/><Stat title="Actor 筛选后有效帧" value={number(t.actor_samples,0)} note={`原始 ${number(t.actor_candidate_samples,0)} · 剔除 ${number(t.actor_filtered_samples,0)}`}/><Stat title="实际优化帧数" value={number(t.actor_optimized_samples,0)} note={t.actor_skip_reason==='no_samples'?'无可用监督帧，本批跳过 Actor':t.actor_skip_reason==='warmup'?'Q/V 预热，Actor 未更新':t.actor_skip_reason==='amp'?'AMP 跳步':'以实际优化器更新为准'}/></div>
      <Trend title="Neutral 有效监督占比" rows={train.rows} series={[{key:'actor_neutral_fraction_before',name:'筛选前',color:'#71aaff',connectNulls:false},{key:'actor_neutral_fraction_after',name:'筛选后',color:'#62d8b6',connectNulls:false}]}/>
    </Card>
    <Card title="最近优化器是否实际更新"><DataTable rows={['actor','critic','value'].map(key=>({id:key,name:key==='critic'?'双 Q':key==='value'?'V':'Actor',updated:t[`${key}_updated`],gradient:t[`${key}_gradient_norm`]}))} columns={[{key:'name',title:'网络'},{key:'updated',title:'实际更新',render:v=>v==null?'未记录':v?'是':'否 / 预热、无样本或 AMP 跳步'},{key:'gradient',title:'梯度范数'}]}/></Card>
    <Card title="训练集完整动作分布 · 全部 144 类" note="全量有效转移统计；不是模型预测频率">{actions.error&&<p className="error">{actions.error}</p>}<DataTable rows={rows} columns={[{key:'id',title:'Joint ID'},{key:'name',title:'完整按键'},{key:'count',title:'样本数'},{key:'fraction',title:'占比',render:v=>percent(v)}]}/></Card>
    <p className="metric-note">历史图显示最近 300 条日志；服务缓存最多 2000 条，完整日志保留在输出目录。当前没有记录同批更新前后概率差，不用猜测值替代。</p>
  </>;
}
