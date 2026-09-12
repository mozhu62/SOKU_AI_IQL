import {Card,DataTable} from '../components/common';
import type {Row} from '../lib/utils';

export function ResourceInputs({usesResources,inputs}:{usesResources?:boolean,inputs?:Row}){
  if(!inputs) return <Card title="技能类型输入"><p className="muted">{usesResources===false?'此模型未声明技能输入，请检查兼容性。':'等待同帧技能类型数据；缺失不显示为默认类型。'}</p></Card>;
  const skills=['self','opponent'].flatMap(side=>(inputs[side]?.skills||[]).map((skill:Row)=>({
    ...skill,side:side==='self'?'己方':'对方',type:skill.valid?['默认','替换 1','替换 2'][skill.variant]:'未知',
  })));
  return <Card title="本次决策使用的技能类型" note={`采集序号 ${inputs.sample_serial} / 游戏帧 ${inputs.battle_frame}；双方四槽，仅 variant + mask`}>
    <DataTable rows={skills} columns={[{key:'side',title:'侧别'},{key:'slot',title:'Skill 槽'},{key:'command',title:'游戏指令'},{key:'type',title:'安装类型'},{key:'valid',title:'有效',render:v=>v?'是':'否'}]}/>
    <p className="muted">技能学习等级、生效等级、手牌、卡槽、选中卡和符卡能量均不再输入网络；模型不输出切卡或用卡。</p>
  </Card>;
}
