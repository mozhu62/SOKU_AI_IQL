import {useEffect,useState} from 'react';
import {command,useQuery,type Row} from '../api';
import {Card,DataTable,ConfirmButton} from '../components/common';
import {number} from '../lib/utils';

const hints:Record<string,string>={batch_size:'每批序列数量；续训锁定',sequence_length:'每序列监督位置数；不是 TCN 感受野',burn_in:'31 帧真实前导历史，固定',replays_per_batch:'每批回放来源抽取次数；不是加载 worker',amp:'CUDA FP16 混合精度',n_step:'TD累计的最大真实转移步数；正整数，越大越占显存',gamma:'未来奖励折扣；不允许在线改变回报定义',expectile:'V 对较高 Q 的偏重程度',advantage_beta:'exp(beta×优势) 的强度，不是探索温度',max_weight:'Actor 优势权重上限',target_tau:'目标 Q 的 EMA 更新比例',actor_lr:'Actor 学习率',critic_lr:'双 Q 学习率',value_lr:'V 学习率',actor_warmup_steps:'该步数前只更新 Q/V',damage_dealt:'对方每扣 1 HP 的奖励系数',damage_taken:'己方每扣 1 HP 的惩罚系数',pressure:'对手防御且灵力下降时的单次事件奖励',far_distance_threshold:'开始持续惩罚的水平距离阈值',far_distance_penalty:'超过距离阈值后每帧扣除的固定值',cache_gb:'解压后的分片内存预算，不是显存',prefetch_batches:'单后台线程的预取队列深度，不增加线程数',cpu_threads:'PyTorch CPU 运算线程，不是 NPZ 加载 worker',validation_batches:'固定种子验证批次数，不代表全量验证'};
export function Parameters({state}:{state:Row}){
  const query=useQuery('parameters',3000),models=useQuery('models',3000),history=useQuery('history?kind=configuration&limit=100',3000);
  const [draft,setDraft]=useState<Row>({}),[error,setError]=useState('');
  const fields:Row[]=query.data?.fields||[],cfg=state.config||{};
  useEffect(()=>{setDraft({});setError('');},[state.stage]);
  const get=(key:string)=>{const [section,name]=key.split('.');return cfg[section]?.[name];};
  const editable=new Set(fields.map(f=>f.key));
  const readonly=Object.entries(cfg).flatMap(([section,value])=>value&&typeof value==='object'?Object.entries(value).filter(([key])=>!editable.has(`${section}.${key}`)).map(([key,v])=>({id:`${section}.${key}`,value:typeof v==='object'?JSON.stringify(v):String(v),hint:hints[key]||'继承配置；结构/数据契约保持不变'})):[]);
  return <><h1 className="page-heading">运行参数与模型文件</h1><p className="notice">先暂停再应用。仅开放不改变损失定义和固定验证口径的运行参数；学习率、奖励、折扣、batch 和输入结构在本页只读。要修改这些值，请另开训练分支。</p>
    <Card title="可在线修改" note={`配置来源：${state.config_source||'准备中'}`}>
      {(error||query.error)&&<p className="error">{error||query.error}</p>}<div className="table-scroll"><table><thead><tr><th>参数</th><th>运行值</th><th>编辑值</th></tr></thead><tbody>{fields.map(f=><tr key={f.key}><td>{f.label}<small className="block">{f.key}</small></td><td>{String(get(f.key)??'未记录')}</td><td><input aria-label={f.label} type="number" min={f.min} max={f.max} step={f.type==='int'?1:.1} disabled={!state.connected||state.state!=='paused'} value={draft[f.key]??get(f.key)??''} onChange={e=>setDraft({...draft,[f.key]:e.target.value})}/></td></tr>)}</tbody></table></div>
      <div className="parameter-actions inline-controls"><ConfirmButton disabled={!state.connected||state.state!=='paused'||!Object.keys(draft).length} title="应用运行参数" description="在更新边界生效，已暂停的预取会被丢弃；不丢失正式训练样本。保存配置快照和当前模型，不改原 YAML。" onConfirm={async()=>{const changes:Row={};for(const [k,v] of Object.entries(draft)){if(String(v).trim()===''||!Number.isFinite(Number(v)))throw new Error('请输入有效数字');changes[k]=Number(v);}const result=await command('apply',{changes});if(result.status==='queued')throw new Error('修改仍在排队，请查看状态，不要重复提交');setDraft({});}}>应用并保存</ConfirmButton><a href="/api/config" download>下载运行配置</a></div>
      <p className="metric-note">在线修改不写原 YAML。恢复修改后的运行参数时，使用输出目录 runtime_config.json，或下载的配置文件作为 --config。</p></Card>
    <Card title="模型与导出策略" note="实战优先下载 actor_bc.pt；last.pt 用于完整 IQL 续训">{models.error&&<p className="error">{models.error}</p>}<DataTable rows={models.data?.rows||[]} columns={[{key:'name',title:'文件'},{key:'kind',title:'用途'},{key:'bytes',title:'MiB',render:v=>number(v/1024**2,1)},{key:'id',title:'操作',render:v=><a href={`/api/models/${v}`} download>下载</a>}]}/></Card>
    <Card title="只读训练定义"><DataTable rows={readonly} columns={[{key:'id',title:'配置项'},{key:'value',title:'运行值'},{key:'hint',title:'说明'}]}/></Card>
    <Card title="参数修改记录">{history.error&&<p className="error">{history.error}</p>}<DataTable rows={history.data?.rows||[]} columns={[{key:'step',title:'Step'},{key:'stage',title:'阶段'},{key:'before',title:'修改前',render:v=>JSON.stringify(v)},{key:'changes',title:'修改后',render:v=>JSON.stringify(v)}]}/></Card>
  </>;
}
